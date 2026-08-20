from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from train_directional_occlusion_proxy_encoder import (
    PoseContrastiveProjectionHead,
    pose_hard_negative_contrastive_loss,
)


class PoseHardNegativeContrastiveLossTest(unittest.TestCase):
    def _loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        loss, _parts = pose_hard_negative_contrastive_loss(
            embeddings,
            logits=torch.tensor([[2.0], [1.5], [1.2], [0.9]]),
            target=torch.tensor([[1.0], [1.0], [0.0], [0.0]]),
            pose_offsets=torch.tensor([0, 4]),
            visible_weights=torch.tensor([[64.0], [16.0], [0.0], [0.0]]),
            temperature=0.1,
            positive_top_k=8,
            negative_top_k=8,
            importance_scale=2.5,
        )
        return loss

    def test_separated_positive_and_negative_sets_have_lower_loss(self) -> None:
        separated = torch.tensor(
            [[1.0, 0.0], [0.98, 0.10], [-1.0, 0.0], [-0.98, -0.10]]
        )
        mixed = torch.tensor(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
        )
        self.assertLess(float(self._loss(separated)), float(self._loss(mixed)))

    def test_loss_is_finite_and_backpropagates(self) -> None:
        embeddings = torch.tensor(
            [[0.8, 0.2], [0.6, 0.4], [-0.4, 0.7], [-0.7, -0.2]],
            requires_grad=True,
        )
        loss = self._loss(embeddings)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertIsNotNone(embeddings.grad)
        self.assertGreater(float(embeddings.grad.abs().sum()), 0.0)

    def test_pose_without_two_positives_returns_differentiable_zero(self) -> None:
        embeddings = torch.randn(3, 4, requires_grad=True)
        loss, parts = pose_hard_negative_contrastive_loss(
            embeddings,
            logits=torch.zeros(3, 1),
            target=torch.tensor([[1.0], [0.0], [0.0]]),
            pose_offsets=torch.tensor([0, 3]),
            visible_weights=torch.ones(3, 1),
            temperature=0.1,
            positive_top_k=8,
            negative_top_k=8,
            importance_scale=2.5,
        )
        loss.backward()
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(parts["contrastiveAnchorCount"], 0.0)

    def test_projection_head_outputs_unit_vectors(self) -> None:
        head = PoseContrastiveProjectionHead(input_dim=6, hidden_dim=8, output_dim=4)
        projected = head(torch.randn(5, 6))
        self.assertEqual(tuple(projected.shape), (5, 4))
        torch.testing.assert_close(
            torch.linalg.norm(projected, dim=-1),
            torch.ones(5),
            atol=1e-5,
            rtol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
