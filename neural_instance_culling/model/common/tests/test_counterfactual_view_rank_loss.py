from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


MODEL_DIR = Path(__file__).resolve().parents[2]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from common.counterfactual_view_rank_loss import (  # noqa: E402
    counterfactual_view_rank_loss,
)


class CounterfactualViewRankLossTests(unittest.TestCase):
    def test_static_instance_bias_cancels(self) -> None:
        logits = torch.tensor([-1.0, 0.5, 0.2, -0.4, 2.0])
        target = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
        ids = torch.tensor([4, 4, 7, 7, 9])
        weights = torch.tensor([8.0, 0.0, 2.0, 0.0, 1.0])
        base = counterfactual_view_rank_loss(logits, target, ids, weights)
        shifted = counterfactual_view_rank_loss(
            logits + torch.tensor([3.0, 3.0, -2.0, -2.0, 7.0]),
            target,
            ids,
            weights,
        )
        torch.testing.assert_close(base[0], shifted[0])
        torch.testing.assert_close(base[1], shifted[1])
        torch.testing.assert_close(
            base[2]["counterfactualViewGap"],
            shifted[2]["counterfactualViewGap"],
        )

    def test_gradients_raise_positive_and_lower_negative_views(self) -> None:
        logits = torch.tensor([-1.0, 1.0, 0.5, -0.5, 2.0], requires_grad=True)
        target = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
        ids = torch.tensor([4, 4, 7, 7, 9])
        weights = torch.tensor([8.0, 0.0, 2.0, 0.0, 1.0])
        positive, negative, parts = counterfactual_view_rank_loss(
            logits, target, ids, weights
        )
        self.assertEqual(parts["counterfactualViewPairCount"], 2.0)
        positive_gradient = torch.autograd.grad(
            positive, logits, retain_graph=True
        )[0]
        negative_gradient = torch.autograd.grad(negative, logits)[0]
        self.assertLess(float(positive_gradient[0]), 0.0)
        self.assertEqual(float(positive_gradient[1]), 0.0)
        self.assertEqual(float(negative_gradient[0]), 0.0)
        self.assertGreater(float(negative_gradient[1]), 0.0)

    def test_visual_mass_limits_tiny_positive_outlier(self) -> None:
        target = torch.tensor([1.0, 1.0, 0.0])
        ids = torch.tensor([3, 3, 3])
        logits = torch.tensor([2.0, -2.0, 0.0])
        high_outlier = counterfactual_view_rank_loss(
            logits, target, ids, torch.tensor([1.0, 100.0, 0.0])
        )
        low_outlier = counterfactual_view_rank_loss(
            logits, target, ids, torch.tensor([100.0, 1.0, 0.0])
        )
        self.assertLess(
            float(low_outlier[0] + low_outlier[1]),
            float(high_outlier[0] + high_outlier[1]),
        )

    def test_priority_can_filter_unregistered_instances(self) -> None:
        logits = torch.tensor([-1.0, 1.0, 0.5, -0.5], requires_grad=True)
        target = torch.tensor([1.0, 0.0, 1.0, 0.0])
        ids = torch.tensor([4, 4, 7, 7])
        weights = torch.tensor([8.0, 0.0, 2.0, 0.0])
        positive, negative, parts = counterfactual_view_rank_loss(
            logits,
            target,
            ids,
            weights,
            recurrence_priority=torch.tensor([0.0, 0.0, 2.0, 2.0]),
        )
        self.assertEqual(parts["counterfactualViewPairCount"], 1.0)
        gradient = torch.autograd.grad(positive + negative, logits)[0]
        torch.testing.assert_close(gradient[:2], torch.zeros(2))
        self.assertLess(float(gradient[2]), 0.0)
        self.assertGreater(float(gradient[3]), 0.0)


if __name__ == "__main__":
    unittest.main()
