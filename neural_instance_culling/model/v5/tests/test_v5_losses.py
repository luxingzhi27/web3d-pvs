from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F


MODEL_DIR = Path(__file__).resolve().parents[2]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from v5.losses import (  # noqa: E402
    COUNT_BUDGET,
    FIELD_NLL_WEIGHT,
    VISUAL_BUDGET,
    SceneDualState,
    constrained_pose_risks,
    external_hit_censor_survival,
    external_hit_event_density,
    external_hit_nll,
    pbce_objective,
    pose_balanced_bce,
)


class ConstrainedRiskTests(unittest.TestCase):
    def test_global_scene_denominators_and_h_terms(self) -> None:
        logits = torch.tensor([[-1.0, 1.0], [0.0, 2.0]])
        targets = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        weights = torch.tensor([[0.0, 2.0], [0.0, 3.0]])
        risks = constrained_pose_risks(
            logits,
            targets,
            weights,
            pose_scale=torch.tensor([1.0, 1.0]),
            G_s=4.0,
            W_s=5.0,
        )
        h_keep = F.softplus(logits[[0, 1], [0, 0]]) / torch.log(torch.tensor(2.0))
        h_miss = F.softplus(-logits[[0, 1], [1, 1]]) / torch.log(torch.tensor(2.0))
        self.assertTrue(torch.allclose(risks.extra, h_keep.sum() / 4.0))
        self.assertTrue(torch.allclose(risks.miss_count, h_miss.sum() / 4.0))
        self.assertTrue(torch.allclose(risks.miss_visual, (h_miss * torch.tensor([2.0, 3.0])).sum() / 5.0))

    def test_pose_scale_is_unbiased_global_scaling_not_batch_normalization(self) -> None:
        logits = torch.tensor([[-2.0, 0.5], [1.5, -0.25]])
        targets = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        weights = torch.tensor([[0.0, 1.0], [0.0, 2.0]])
        # Sampling one of two poses gives inverse inclusion scale 2.  The
        # result is still divided by the full scene G_s/W_s.
        selected = constrained_pose_risks(
            logits[:1],
            targets[:1],
            weights[:1],
            pose_scale=2.0,
            G_s=4.0,
            W_s=3.0,
        )
        expected_extra = 2.0 * F.softplus(logits[0, 0]) / torch.log(torch.tensor(2.0)) / 4.0
        expected_count = 2.0 * F.softplus(-logits[0, 1]) / torch.log(torch.tensor(2.0)) / 4.0
        expected_visual = 2.0 * F.softplus(-logits[0, 1]) / torch.log(torch.tensor(2.0)) / 3.0
        self.assertTrue(torch.allclose(selected.extra, expected_extra))
        self.assertTrue(torch.allclose(selected.miss_count, expected_count))
        self.assertTrue(torch.allclose(selected.miss_visual, expected_visual))


class SceneDualStateTests(unittest.TestCase):
    def test_loss_update_and_checkpoint_state(self) -> None:
        logits = torch.tensor([[-1.0, 0.5]], requires_grad=True)
        risks = constrained_pose_risks(
            logits,
            torch.tensor([[0.0, 1.0]]),
            torch.tensor([[0.0, 2.0]]),
            pose_scale=1.0,
            G_s=2.0,
            W_s=2.0,
        )
        state = SceneDualState(["scene"], dual_lr=2.0, device="cpu")
        state.lambda_count[0] = 0.3
        state.lambda_visual[0] = 0.4
        field = torch.tensor(0.7, requires_grad=True)
        loss = state.loss("scene", risks, field)
        expected = risks.extra + 0.3 * risks.miss_count + 0.4 * risks.miss_visual + FIELD_NLL_WEIGHT * field
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(field.grad)

        state.update("scene", risks)
        self.assertAlmostEqual(
            float(state.lambda_count[0]),
            max(0.0, 0.3 + 2.0 * (float(risks.miss_count) - COUNT_BUDGET)),
        )
        self.assertAlmostEqual(
            float(state.lambda_visual[0]),
            max(0.0, 0.4 + 2.0 * (float(risks.miss_visual) - VISUAL_BUDGET)),
        )
        checkpoint = state.state_dict()
        restored = SceneDualState(["scene"], dual_lr=0.1)
        restored.load_state_dict(checkpoint)
        self.assertTrue(torch.equal(restored.lambda_count, state.lambda_count))
        self.assertTrue(torch.equal(restored.lambda_visual, state.lambda_visual))
        self.assertEqual(restored.dual_lr, state.dual_lr)


class CurrentStatusFieldTests(unittest.TestCase):
    def test_thirteen_distance_grid_uses_current_status_bernoulli_nll(self) -> None:
        torch.manual_seed(7)
        coefficients = torch.randn(2, 4, 7, dtype=torch.float64, requires_grad=True)
        directions = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
        distances = torch.linspace(0.25, 3.25, 13, dtype=torch.float64).repeat(2, 1)
        radii = torch.tensor([1.0, 2.0], dtype=torch.float64)
        events = torch.zeros(2, 13, dtype=torch.float64)
        events[:, 4:] = 1.0
        loss = external_hit_nll(coefficients, directions, distances, radii, events)
        survival = external_hit_censor_survival(coefficients, directions, distances, radii)
        expected = -(events * torch.log1p(-survival.clamp(max=1.0 - 1.0e-15)) + (1.0 - events) * torch.log(survival)).mean()
        self.assertTrue(torch.allclose(loss, expected, atol=1.0e-10, rtol=1.0e-8))
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertIsNotNone(coefficients.grad)
        self.assertTrue(bool(torch.isfinite(coefficients.grad).all()))

    def test_density_diagnostic_is_not_the_current_status_loss(self) -> None:
        coefficients = torch.randn(1, 4, 7, dtype=torch.float64, requires_grad=True)
        directions = torch.tensor([[0.3, -0.4, 0.5]], dtype=torch.float64)
        distances = torch.tensor([[1.2]], dtype=torch.float64)
        radii = torch.tensor([1.7], dtype=torch.float64)
        density = external_hit_event_density(coefficients, directions, distances, radii)
        survival = external_hit_censor_survival(coefficients, directions, distances, radii)
        self.assertTrue(bool(torch.isfinite(density).all()))
        self.assertTrue(bool(torch.isfinite(survival).all()))
        self.assertTrue(bool((density > 0.0).all()))


class PbceTests(unittest.TestCase):
    def test_empty_positive_and_empty_negative_poses_are_finite(self) -> None:
        logits = torch.tensor([-1.0, 0.5, 2.0, -0.25], requires_grad=True)
        targets = torch.tensor([0.0, 0.0, 1.0, 1.0])
        pose_ids = torch.tensor([0, 0, 1, 1])
        loss = pose_balanced_bce(logits, targets, pose_ids, num_poses=3)
        expected = torch.stack([
            F.softplus(logits[:2]).mean(),
            F.softplus(-logits[2:]).mean(),
        ]).mean()
        self.assertTrue(torch.allclose(loss, expected))
        total = pbce_objective(logits, targets, pose_ids, 3, torch.tensor(0.4))
        self.assertTrue(torch.allclose(total, loss + FIELD_NLL_WEIGHT * 0.4))
        total.backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))

    def test_offsets_and_ranges_are_accepted(self) -> None:
        logits = torch.tensor([0.0, 1.0, -1.0])
        targets = torch.tensor([0.0, 1.0, 0.0])
        expected = pose_balanced_bce(logits, targets, torch.tensor([0, 0, 1]), 2)
        offsets = pose_balanced_bce(logits, targets, torch.tensor([0, 2, 3]), 2)
        ranges = pose_balanced_bce(logits, targets, torch.tensor([[0, 2], [2, 3]]), 2)
        self.assertTrue(torch.allclose(expected, offsets))
        self.assertTrue(torch.allclose(expected, ranges))


if __name__ == "__main__":
    unittest.main()
