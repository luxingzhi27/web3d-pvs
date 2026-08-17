from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

MODEL = Path(__file__).resolve().parents[2]
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

from common.safety_reserve_operating_utility_loss import (  # noqa: E402
    OPERATING_THRESHOLD_ANCHORS,
    _group_topk_smooth_max,
    bounded_topk_smooth_max,
    project_operating_utility_gradient_groups,
    safety_reserve_gate,
    safety_reserve_operating_utility_loss,
    sample_train_operating_thresholds,
    soft_request_probability,
)


class SafetyReserveOperatingUtilityLossTest(unittest.TestCase):
    @staticmethod
    def _inputs(*, requires_grad: bool = False) -> tuple[torch.Tensor, ...]:
        logits = torch.tensor(
            [1.8, -2.0, -0.8, 1.2, -1.5, -0.3],
            dtype=torch.float32,
            requires_grad=requires_grad,
        )
        target = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        pose_offsets = torch.tensor([0, 3, 6], dtype=torch.long)
        visible_weights = torch.tensor([80.0, 0.0, 0.0, 20.0, 0.0, 0.0])
        visible_hit_rates = torch.tensor([0.25, 0.0, 0.0, 0.81, 0.0, 0.0])
        instance_ids = torch.tensor([7, 2, 9, 5, 11, 3], dtype=torch.long)
        instance_to_glb = torch.tensor([0, 0, 1, 3, 1, 2, 2, 0, 3, 1, 2, 3])
        glb_bytes = torch.tensor([1_000.0, 20_000.0, 4_000.0, 8_000.0])
        return (
            logits,
            target,
            pose_offsets,
            visible_weights,
            visible_hit_rates,
            instance_ids,
            instance_to_glb,
            glb_bytes,
        )

    def test_threshold_sampler_is_train_only_deterministic_and_anchored(self) -> None:
        first = sample_train_operating_thresholds(20260801, 8)
        repeated = sample_train_operating_thresholds(20260801, 8)
        changed = sample_train_operating_thresholds(20260801, 9)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first[:2], changed[:2])
        self.assertEqual(len(first), 3)
        self.assertIn(first[-1], OPERATING_THRESHOLD_ANCHORS)
        self.assertTrue(all(1e-4 <= value <= 0.2 for value in first))
        with self.assertRaisesRegex(ValueError, "train-only"):
            sample_train_operating_thresholds(20260801, 8, split="calibration")
        with self.assertRaisesRegex(ValueError, "train-only"):
            sample_train_operating_thresholds(20260801, 8, split="test")

    def test_bounded_smooth_max_does_not_saturate_when_low_logits_are_copied(self) -> None:
        single = bounded_topk_smooth_max(torch.tensor([-8.0]))
        copied = bounded_topk_smooth_max(torch.full((512,), -8.0))
        self.assertTrue(torch.allclose(single, copied, atol=1e-7, rtol=0.0))
        request_single = soft_request_probability(single, 0.02)
        request_copied = soft_request_probability(copied, 0.02)
        self.assertTrue(torch.allclose(request_single, request_copied, atol=1e-9, rtol=0.0))
        self.assertLess(float(request_copied), 1e-8)

    def test_highest_group_logit_monotonically_increases_request(self) -> None:
        low = bounded_topk_smooth_max(torch.tensor([-5.0, -4.0, -6.0]))
        high = bounded_topk_smooth_max(torch.tensor([-5.0, -2.0, -6.0]))
        self.assertGreater(float(high), float(low))
        self.assertGreater(
            float(soft_request_probability(high, 0.02)),
            float(soft_request_probability(low, 0.02)),
        )

    def test_group_topk_smooth_max_matches_independent_bounded_groups(self) -> None:
        logits = torch.tensor([1.0, 4.0, 2.0, -3.0, 0.5, 2.5, -1.0])
        groups = torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
        actual = _group_topk_smooth_max(logits, groups, 2, top_k=2, temperature=0.2)
        expected = torch.stack(
            [
                bounded_topk_smooth_max(logits[groups == group], top_k=2, temperature=0.2)
                for group in range(2)
            ]
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_warmup_forces_zero_detached_gate_and_efficiency_gradient(self) -> None:
        positive_logits = torch.tensor([1.0, 2.0], requires_grad=True)
        gate, _margin = safety_reserve_gate(
            positive_logits,
            torch.tensor([1.0, 0.5]),
            0.02,
            global_step=9,
            total_optimizer_steps=100,
        )
        self.assertEqual(float(gate), 0.0)
        self.assertFalse(gate.requires_grad)

        inputs = self._inputs(requires_grad=True)
        _total, parts = safety_reserve_operating_utility_loss(
            *inputs,
            train_seed=20260801,
            global_step=0,
            total_optimizer_steps=100,
        )
        efficiency = torch.as_tensor(parts["lossEfficiency"])
        gradient = torch.autograd.grad(efficiency, inputs[0], retain_graph=True)[0]
        self.assertEqual(float(efficiency), 0.0)
        self.assertTrue(torch.equal(gradient, torch.zeros_like(gradient)))

    def test_reserve_gate_opens_monotonically_and_is_stop_gradient(self) -> None:
        weights = torch.tensor([1.0, 0.5])
        low_gate, low_margin = safety_reserve_gate(
            torch.tensor([-4.5, -3.5], requires_grad=True),
            weights,
            0.02,
            global_step=10,
            total_optimizer_steps=100,
        )
        high_gate, high_margin = safety_reserve_gate(
            torch.tensor([-2.5, -1.5], requires_grad=True),
            weights,
            0.02,
            global_step=10,
            total_optimizer_steps=100,
        )
        self.assertGreater(float(high_margin), float(low_margin))
        self.assertGreater(float(high_gate), float(low_gate))
        self.assertFalse(low_gate.requires_grad)
        self.assertFalse(high_gate.requires_grad)

    def test_reserve_margin_is_finite_and_diagnostic_only(self) -> None:
        logits = torch.tensor([-3.0, -1.0, 0.5], requires_grad=True)
        _gate, margin = safety_reserve_gate(
            logits,
            torch.tensor([1.0, 2.0, 1.0]),
            0.02,
            global_step=20,
            total_optimizer_steps=100,
        )
        self.assertTrue(bool(torch.isfinite(margin)))
        self.assertFalse(margin.requires_grad)

    def test_all_auxiliary_gradient_dots_are_non_negative_after_projection(self) -> None:
        safety = [torch.tensor([2.0, 0.0]), None]
        relation = [torch.tensor([-3.0, 1.0]), None]
        schedule = [torch.tensor([1.0, 1.0]), None]
        efficiency = [torch.tensor([-4.0, 2.0]), None]
        projected, statistics = project_operating_utility_gradient_groups(
            safety,
            relation,
            schedule,
            efficiency,
            relation_norm_cap=10.0,
            schedule_norm_cap=10.0,
            efficiency_norm_cap=0.25,
        )
        for name in ("relation", "schedule", "efficiency"):
            self.assertGreaterEqual(
                float(torch.dot(safety[0], projected[name][0])),
                -1e-6,
                msg=name,
            )
        self.assertLessEqual(
            float(projected["efficiency"][0].norm()),
            0.25 * float(safety[0].norm()) + 1e-6,
        )
        self.assertTrue(all(math.isfinite(value) for value in statistics.values()))
        self.assertEqual(statistics["relationGradientProjectionApplied"], 1.0)
        self.assertEqual(statistics["scheduleGradientProjectionApplied"], 0.0)

    def test_projection_remains_non_negative_under_large_fp32_cancellation(self) -> None:
        generator = torch.Generator().manual_seed(20260815)
        safety_value = torch.randn(200000, generator=generator)
        auxiliary = torch.randn(200000, generator=generator)
        auxiliary = auxiliary - (
            torch.dot(safety_value, auxiliary) / torch.dot(safety_value, safety_value)
        ) * safety_value
        auxiliary = auxiliary - 1e-6 * safety_value
        projected, statistics = project_operating_utility_gradient_groups(
            [safety_value],
            [auxiliary],
            [torch.zeros_like(auxiliary)],
            [torch.zeros_like(auxiliary)],
            relation_norm_cap=10.0,
            schedule_norm_cap=10.0,
            efficiency_norm_cap=10.0,
        )
        self.assertGreaterEqual(statistics["relationGradientDotAfterProjection"], 0.0)
        self.assertGreaterEqual(float(torch.dot(safety_value, projected["relation"][0])), -1e-4)

    def test_loss_is_finite_and_objective_parts_sum_exactly(self) -> None:
        inputs = self._inputs(requires_grad=True)
        total, parts = safety_reserve_operating_utility_loss(
            *inputs,
            train_seed=20260801,
            global_step=50,
            total_optimizer_steps=100,
            boundary_tail_fraction=0.5,
            negative_tail_fraction=0.5,
        )
        expected = (
            torch.as_tensor(parts["lossSafetyRvl"])
            + torch.as_tensor(parts["lossBoundaryTailScaled"])
            + torch.as_tensor(parts["lossEfficiency"])
        )
        self.assertTrue(torch.allclose(total, expected, atol=1e-7, rtol=1e-7))
        self.assertTrue(torch.allclose(total, torch.as_tensor(parts["lossSafetyReserveOperatingUtility"])))
        for name, value in parts.items():
            if isinstance(value, torch.Tensor):
                self.assertTrue(bool(torch.isfinite(value).all()), msg=name)
        total.backward()
        self.assertIsNotNone(inputs[0].grad)
        self.assertTrue(bool(torch.isfinite(inputs[0].grad).all()))

    def test_loss_rejects_non_train_split(self) -> None:
        with self.assertRaisesRegex(ValueError, "train-only"):
            safety_reserve_operating_utility_loss(
                *self._inputs(),
                train_seed=20260801,
                global_step=0,
                total_optimizer_steps=100,
                split="validation",
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_one_step_cuda_loss_and_backward_are_finite(self) -> None:
        cpu_inputs = self._inputs(requires_grad=True)
        inputs = (
            cpu_inputs[0].cuda().detach().requires_grad_(),
            *(value.cuda() for value in cpu_inputs[1:]),
        )
        total, parts = safety_reserve_operating_utility_loss(
            *inputs,
            train_seed=20260801,
            global_step=1,
            total_optimizer_steps=4,
        )
        total.backward()
        self.assertTrue(bool(torch.isfinite(total)))
        self.assertTrue(bool(torch.isfinite(inputs[0].grad).all()))
        self.assertTrue(bool(torch.isfinite(torch.as_tensor(parts["noDemandGlbCount"]))))


if __name__ == "__main__":
    unittest.main()
