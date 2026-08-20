from __future__ import annotations

import unittest

import torch

from neural_instance_culling.model.common.pose_balanced_rvl_contrastive_loss import (
    TrainingOnlyContrastiveProjectionHead,
    pose_balanced_rvl_contrastive_visibility_loss,
    pose_weighted_recall_guard_loss,
    shared_tail_logit_contrastive_loss,
)


class PoseBalancedRvlContrastiveLossTests(unittest.TestCase):
    def test_recall_guard_uses_visible_weight_mass(self) -> None:
        labels = torch.tensor([1.0, 1.0, 0.0])
        offsets = torch.tensor([0, 3])
        weights = torch.tensor([99.0, 1.0, 0.0])
        important_missed = torch.tensor([-3.0, 3.0, -3.0])
        minor_missed = torch.tensor([3.0, -3.0, -3.0])

        important_loss, important_parts = pose_weighted_recall_guard_loss(
            important_missed, labels, offsets, weights
        )
        minor_loss, _ = pose_weighted_recall_guard_loss(
            minor_missed, labels, offsets, weights
        )

        self.assertGreater(float(important_loss), float(minor_loss))
        self.assertLess(
            float(important_parts["softAggregateWeightedRecall"]),
            0.1,
        )

    def test_recall_guard_has_no_negative_gradient(self) -> None:
        logits = torch.tensor([-1.0, 2.0, 3.0], requires_grad=True)
        loss, _ = pose_weighted_recall_guard_loss(
            logits,
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([0, 3]),
            torch.tensor([1.0, 0.0, 0.0]),
        )
        loss.backward()

        self.assertLess(float(logits.grad[0]), 0.0)
        self.assertEqual(float(logits.grad[1]), 0.0)
        self.assertEqual(float(logits.grad[2]), 0.0)

    def test_shared_tail_uses_only_current_margin_violations(self) -> None:
        logits = torch.tensor([-1.0, 2.0, 1.0, -3.0], requires_grad=True)
        embeddings = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7], [0.0, -1.0]],
            requires_grad=True,
        )
        loss, parts = shared_tail_logit_contrastive_loss(
            logits,
            embeddings,
            torch.tensor([1.0, 1.0, 0.0, 0.0]),
            torch.tensor([0, 4]),
            torch.tensor([1.0, 1.0, 0.0, 0.0]),
            positive_mass_fraction=0.5,
            negative_top_fraction=0.5,
            margin=0.5,
        )
        loss.backward()

        self.assertGreater(float(loss), 0.0)
        self.assertEqual(parts["sharedTailActivePairCount"], 1.0)
        self.assertLess(float(logits.grad[0]), 0.0)
        self.assertGreater(float(logits.grad[2]), 0.0)
        self.assertEqual(float(logits.grad[1]), 0.0)
        self.assertEqual(float(logits.grad[3]), 0.0)
        self.assertGreater(float(embeddings.grad.abs().sum()), 0.0)

    def test_shared_tail_stops_after_the_margin_is_satisfied(self) -> None:
        logits = torch.tensor([3.0, 2.0, -2.0, -3.0], requires_grad=True)
        embeddings = torch.randn(4, 3, requires_grad=True)
        loss, parts = shared_tail_logit_contrastive_loss(
            logits,
            embeddings,
            torch.tensor([1.0, 1.0, 0.0, 0.0]),
            torch.tensor([0, 4]),
            torch.tensor([1.0, 1.0, 0.0, 0.0]),
            positive_mass_fraction=0.5,
            negative_top_fraction=0.5,
            margin=0.5,
        )
        loss.backward()

        self.assertEqual(float(loss), 0.0)
        self.assertEqual(parts["sharedTailActivePairCount"], 0.0)
        self.assertEqual(float(logits.grad.abs().sum()), 0.0)
        self.assertEqual(float(embeddings.grad.abs().sum()), 0.0)

    def test_complete_loss_has_nonduplicated_registered_terms(self) -> None:
        logits = torch.tensor([-0.5, 0.2, 0.7, -0.4], requires_grad=True)
        embeddings = torch.randn(4, 5, requires_grad=True)
        loss, parts = pose_balanced_rvl_contrastive_visibility_loss(
            logits,
            embeddings,
            torch.tensor([1.0, 0.0, 1.0, 0.0]),
            torch.tensor([0, 2, 4]),
            torch.tensor([4.0, 0.0, 1.0, 0.0]),
            recall_guard_weight=0.3,
            separation_weight=0.2,
        )

        torch.testing.assert_close(
            loss,
            parts["lossVisibilityPoseBalanced"]
            + parts["lossVisibilityRvlRecallGuardWeighted"]
            + parts["lossVisibilitySharedTailWeighted"],
        )
        self.assertNotIn("lossTversky", parts)
        self.assertNotIn("lossCount", parts)
        self.assertNotIn("lossBce", parts)

    def test_projection_head_is_unit_normalized(self) -> None:
        head = TrainingOnlyContrastiveProjectionHead(6, 8, 4)
        output = head(torch.randn(5, 6))
        torch.testing.assert_close(
            torch.linalg.norm(output, dim=-1),
            torch.ones(5),
            atol=1e-5,
            rtol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
