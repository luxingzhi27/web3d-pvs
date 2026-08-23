from __future__ import annotations

import unittest

import torch

from neural_instance_culling.model.common.dual_probe_rescue_loss import (
    dual_probe_rescue_attenuation_loss,
)


class DualProbeRescueLossTests(unittest.TestCase):
    def test_negative_candidates_receive_decay_gradient_and_frozen_inputs_are_detached(self) -> None:
        attenuation = torch.tensor(
            [[0.7, 0.6], [0.8, 0.5], [0.4, 0.3]],
            requires_grad=True,
        )
        base = torch.tensor([-2.0, 1.0, 2.0], requires_grad=True)
        primary_gate = torch.tensor([1.0, 0.0, 0.0], requires_grad=True)
        coverage_gate = torch.tensor([0.0, 1.0, 0.0], requires_grad=True)
        target = torch.tensor([1.0, 1.0, 0.0])
        visible_weights = torch.tensor([9.0, 1.0, 0.0])
        loss, diagnostics = dual_probe_rescue_attenuation_loss(
            attenuation,
            base,
            primary_gate,
            coverage_gate,
            target,
            visible_weights,
            fixed_residual=torch.tensor([0.5, 0.5, 0.0]),
            residual=torch.tensor([0.4, 0.3, 0.0]),
        )
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss).all()))
        self.assertGreater(float(attenuation.grad[2].mean()), 0.0)
        self.assertLess(float(attenuation.grad[0, 0]), 0.0)
        self.assertIsNone(base.grad)
        self.assertIsNone(primary_gate.grad)
        self.assertIsNone(coverage_gate.grad)
        self.assertTrue(diagnostics["baseLogitDetached"])
        self.assertTrue(diagnostics["frozenGatesDetached"])
        self.assertTrue(diagnostics["classNormalized"])

    def test_primary_tail_uses_sqrt_visual_weight_and_coverage_tail_is_equal_weighted(self) -> None:
        attenuation = torch.zeros((4, 2), requires_grad=True)
        base = torch.zeros(4)
        primary_gate = torch.ones(4)
        coverage_gate = torch.ones(4)
        target = torch.ones(4)
        visible_weights = torch.tensor([1.0, 9.0, 1.0, 9.0])
        loss, diagnostics = dual_probe_rescue_attenuation_loss(
            attenuation,
            base,
            primary_gate,
            coverage_gate,
            target,
            visible_weights,
            primary_low_score_positive=torch.tensor([1, 1, 0, 0]),
            coverage_low_score_positive=torch.tensor([0, 0, 1, 1]),
            negative_decay_weight=0.0,
            high_negative_decay_weight=0.0,
        )
        loss.backward()
        primary_ratio = abs(float(attenuation.grad[1, 0] / attenuation.grad[0, 0]))
        coverage_ratio = abs(float(attenuation.grad[3, 1] / attenuation.grad[2, 1]))
        self.assertAlmostEqual(primary_ratio, 3.0, places=5)
        self.assertAlmostEqual(coverage_ratio, 1.0, places=5)
        self.assertEqual(diagnostics["primaryLowScorePositiveCount"], 2.0)
        self.assertEqual(diagnostics["coverageLowScorePositiveCount"], 2.0)

    def test_shape_finite_and_posterior_bound_validation(self) -> None:
        common = {
            "base_logit": torch.zeros(2),
            "primary_gate": torch.ones(2),
            "coverage_gate": torch.ones(2),
            "target": torch.tensor([1.0, 0.0]),
            "visible_weights": torch.ones(2),
        }
        with self.assertRaisesRegex(ValueError, "shape \[B, 2\]"):
            dual_probe_rescue_attenuation_loss(torch.zeros(2), **common)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            dual_probe_rescue_attenuation_loss(
                torch.ones((2, 2)),
                **{**common, "primary_gate": torch.tensor([-1.0, 0.0])},
            )
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            dual_probe_rescue_attenuation_loss(
                torch.ones((2, 2)),
                **common,
                fixed_residual=torch.tensor([0.5, 0.5]),
                residual=torch.tensor([0.6, 0.1]),
            )


if __name__ == "__main__":
    unittest.main()
