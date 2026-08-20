from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

MODEL = Path(__file__).resolve().parents[2]
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

from common.safety_reserve_operating_utility_loss import (  # noqa: E402
    OPERATING_THRESHOLD_ANCHORS,
    _group_topk_smooth_max,
    bounded_topk_smooth_max,
    candidate_boundary_hard_negative_loss,
    cull_certificate_pose_tail_pair_loss,
    cull_certificate_tail_classification_loss,
    extreme_tail_separation_loss,
    instance_exposure_balanced_bce_loss,
    project_operating_utility_gradient_groups,
    safety_boundary_negative_excess_loss,
    safety_reserve_gate,
    safety_reserve_operating_utility_loss,
    same_instance_cross_view_rank_loss,
    sample_train_operating_thresholds,
    soft_request_probability,
    view_residual_tail_regularizers,
    weighted_positive_safety_boundary_logit,
    weighted_positive_tail_compactness_loss,
)
from common.safety_constraint_utility_loss import rvl_strong_v2_visibility_loss  # noqa: E402


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

    def test_candidate_boundary_loss_only_updates_selected_negative(self) -> None:
        logits = torch.tensor([0.0, -1.0, -1.0], requires_grad=True)
        target = torch.tensor([1.0, 0.0, 0.0])
        loss, parts = candidate_boundary_hard_negative_loss(
            logits,
            target,
            torch.tensor([0, 3]),
            torch.tensor([10.0, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0]),
            negative_tail_fraction=0.5,
            selection_bias=1.0,
            proximity_gain=1.0,
        )
        loss.backward()
        assert logits.grad is not None
        self.assertEqual(float(logits.grad[0]), 0.0)
        self.assertGreater(float(logits.grad[1]), 0.0)
        self.assertEqual(float(logits.grad[2]), 0.0)
        self.assertEqual(parts["candidateBoundarySelectedNegativeCount"], 1.0)
        self.assertAlmostEqual(
            float(parts["candidateBoundarySelectedProximity"]), 1.0
        )

    def test_candidate_boundary_proximity_gain_is_finite(self) -> None:
        loss, parts = candidate_boundary_hard_negative_loss(
            torch.tensor([0.0, -0.5]),
            torch.tensor([1.0, 0.0]),
            torch.tensor([0, 2]),
            torch.tensor([5.0, 0.0]),
            torch.tensor([0.0, 1.0]),
            proximity_gain=2.0,
        )
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertAlmostEqual(
            float(parts["candidateBoundarySelectedProximity"]), 1.0
        )

    def test_view_residual_regularizers_guard_positives_and_reject_global_shift(self) -> None:
        final = torch.tensor([-2.5, -1.0, -5.0], requires_grad=True)
        base = torch.tensor([-2.0, -1.0, -5.0])
        residual = torch.tensor([-0.5, 0.25, 0.25], requires_grad=True)
        guard, anchor, center, parts = view_residual_tail_regularizers(
            final,
            base,
            residual,
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([10.0, 0.0, 0.0]),
        )

        guard_gradient = torch.autograd.grad(guard, final)[0]
        self.assertLess(float(guard_gradient[0]), 0.0)
        self.assertEqual(float(guard_gradient[1:].abs().sum()), 0.0)
        self.assertGreater(float(anchor), 0.0)
        self.assertEqual(float(center), 0.0)
        self.assertEqual(float(parts["viewResidualPositiveDropFraction"]), 1.0)

        shifted = torch.ones(4, requires_grad=True)
        _guard, shifted_anchor, shifted_center, _parts = view_residual_tail_regularizers(
            torch.zeros(4),
            torch.zeros(4),
            shifted,
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
        )
        self.assertEqual(float(shifted_anchor), 1.0)
        self.assertEqual(float(shifted_center), 1.0)

    def test_view_residual_positive_guard_can_protect_low_weight_positives(self) -> None:
        final = torch.tensor([-2.0, -2.0], requires_grad=True)
        base = torch.tensor([-1.0, -1.0])
        target = torch.ones(2)
        weights = torch.tensor([100.0, 1.0])
        residual = torch.tensor([-1.0, -1.0])

        uniform_guard, *_ = view_residual_tail_regularizers(
            final,
            base,
            residual,
            target,
            weights,
            positive_importance_mix=0.0,
        )
        weighted_guard, *_ = view_residual_tail_regularizers(
            final,
            base,
            residual,
            target,
            weights,
            positive_importance_mix=1.0,
        )
        uniform_gradient = torch.autograd.grad(
            uniform_guard, final, retain_graph=True
        )[0]
        weighted_gradient = torch.autograd.grad(weighted_guard, final)[0]

        self.assertAlmostEqual(float(uniform_gradient[0]), float(uniform_gradient[1]))
        self.assertGreater(
            abs(float(weighted_gradient[0])), abs(float(weighted_gradient[1]))
        )

        with self.assertRaisesRegex(ValueError, "importance mix"):
            view_residual_tail_regularizers(
                final,
                base,
                residual,
                target,
                weights,
                positive_importance_mix=1.1,
            )

    def test_same_instance_cross_view_rank_requires_view_dependent_separation(self) -> None:
        logits = torch.tensor(
            [-1.0, 1.0, 0.5, -0.5, 2.0], requires_grad=True
        )
        target = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
        instance_ids = torch.tensor([4, 4, 7, 7, 9])
        weights = torch.tensor([8.0, 0.0, 2.0, 0.0, 1.0])
        positive_loss, negative_loss, parts = same_instance_cross_view_rank_loss(
            logits,
            target,
            instance_ids,
            weights,
            margin=0.5,
            temperature=0.25,
        )
        self.assertEqual(parts["sameInstanceCrossViewPairCount"], 2.0)
        positive_gradient = torch.autograd.grad(
            positive_loss, logits, retain_graph=True
        )[0]
        negative_gradient = torch.autograd.grad(negative_loss, logits)[0]
        self.assertLess(float(positive_gradient[0]), 0.0)
        self.assertEqual(float(positive_gradient[1]), 0.0)
        self.assertEqual(float(negative_gradient[0]), 0.0)
        self.assertGreater(float(negative_gradient[1]), 0.0)
        separated_positive, separated_negative, _ = same_instance_cross_view_rank_loss(
            torch.tensor([2.0, -2.0, 1.5, -1.5, 2.0]),
            target,
            instance_ids,
            weights,
            margin=0.5,
            temperature=0.25,
        )
        self.assertLess(
            float(separated_positive + separated_negative),
            float(positive_loss.detach() + negative_loss.detach()),
        )

    def test_same_instance_cross_view_rank_filters_by_recurrence_priority(self) -> None:
        logits = torch.tensor([-1.0, 1.0, 0.5, -0.5], requires_grad=True)
        target = torch.tensor([1.0, 0.0, 1.0, 0.0])
        instance_ids = torch.tensor([4, 4, 7, 7])
        weights = torch.tensor([8.0, 0.0, 2.0, 0.0])
        priority = torch.tensor([0.0, 0.0, 2.0, 2.0])

        positive_loss, negative_loss, parts = same_instance_cross_view_rank_loss(
            logits,
            target,
            instance_ids,
            weights,
            recurrence_priority=priority,
            margin=0.5,
            temperature=0.25,
        )
        self.assertEqual(parts["sameInstanceCrossViewPairCount"], 1.0)
        gradient = torch.autograd.grad(positive_loss + negative_loss, logits)[0]
        torch.testing.assert_close(gradient[:2], torch.zeros(2))
        self.assertLess(float(gradient[2]), 0.0)
        self.assertGreater(float(gradient[3]), 0.0)

        zero_positive, zero_negative, zero_parts = same_instance_cross_view_rank_loss(
            logits,
            target,
            instance_ids,
            weights,
            recurrence_priority=torch.zeros_like(priority),
        )
        self.assertEqual(zero_parts["sameInstanceCrossViewPairCount"], 0.0)
        torch.testing.assert_close(
            torch.autograd.grad(zero_positive + zero_negative, logits)[0],
            torch.zeros_like(logits),
        )

    def test_instance_exposure_bce_routes_class_gradients_separately(self) -> None:
        logits = torch.tensor([-1.0, 1.0, 0.5, -0.5], requires_grad=True)
        target = torch.tensor([1.0, 0.0, 1.0, 0.0])
        ids = torch.tensor([0, 0, 1, 1])
        positive_table = torch.tensor([4.0, 1.0])
        negative_table = torch.tensor([1.0, 3.0])
        positive_loss, negative_loss, parts = instance_exposure_balanced_bce_loss(
            logits, target, ids, positive_table, negative_table
        )
        positive_gradient = torch.autograd.grad(
            positive_loss, logits, retain_graph=True
        )[0]
        negative_gradient = torch.autograd.grad(negative_loss, logits)[0]
        self.assertTrue(bool((positive_gradient[target > 0.5] < 0.0).all()))
        self.assertTrue(bool((positive_gradient[target <= 0.5] == 0.0).all()))
        self.assertTrue(bool((negative_gradient[target <= 0.5] > 0.0).all()))
        self.assertTrue(bool((negative_gradient[target > 0.5] == 0.0).all()))
        self.assertEqual(parts["instanceExposurePositiveCount"], 2.0)
        self.assertEqual(parts["instanceExposureNegativeCount"], 2.0)

    def test_rvl_diagnostic_controls_change_only_requested_terms(self) -> None:
        logits = torch.zeros(4)
        target = torch.tensor([1.0, 0.0, 0.0, 0.0])
        offsets = torch.tensor([0, 4], dtype=torch.long)
        weights = torch.ones(4)
        evidence = torch.zeros(4)
        default_loss, default_parts = rvl_strong_v2_visibility_loss(
            logits, target, offsets, weights, evidence
        )
        no_count_loss, no_count_parts = rvl_strong_v2_visibility_loss(
            logits, target, offsets, weights, evidence, count_weight=0.0
        )
        negative_normalized_loss, negative_parts = rvl_strong_v2_visibility_loss(
            logits,
            target,
            offsets,
            weights,
            evidence,
            fp_normalization="negative",
        )
        self.assertTrue(torch.allclose(default_parts["lossCount"], no_count_parts["lossCount"]))
        self.assertTrue(
            torch.allclose(
                default_loss - no_count_loss,
                0.10 * default_parts["lossCount"],
            )
        )
        self.assertGreater(float(default_parts["lossRvlFp"]), float(negative_parts["lossRvlFp"]))
        self.assertLess(float(negative_normalized_loss), float(default_loss))

        focused_rank_loss, focused_rank_parts = rvl_strong_v2_visibility_loss(
            logits,
            target,
            offsets,
            weights,
            evidence,
            rank_weight=0.90,
            rank_negative_top_k=1,
        )
        self.assertTrue(torch.allclose(default_parts["lossRank"], focused_rank_parts["lossRank"]))
        self.assertGreater(float(focused_rank_loss), float(default_loss))

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

    def test_softplus_negative_band_avoids_high_score_sigmoid_saturation(self) -> None:
        inputs = list(self._inputs(requires_grad=False))
        inputs[0] = torch.zeros_like(inputs[0], requires_grad=True)
        _sigmoid_total, sigmoid_parts = safety_reserve_operating_utility_loss(
            *inputs,
            train_seed=20260801,
            global_step=0,
            total_optimizer_steps=10,
            boundary_tail_weight=0.0,
            negative_band_weight=1.0,
            glb_resource_weight=0.0,
            warmup_fraction=0.0,
            negative_band_shape="sigmoid",
        )
        sigmoid_gradient = torch.autograd.grad(
            torch.as_tensor(sigmoid_parts["lossEfficiency"]), inputs[0], retain_graph=False
        )[0]

        softplus_inputs = list(self._inputs(requires_grad=False))
        softplus_inputs[0] = torch.zeros_like(softplus_inputs[0], requires_grad=True)
        _softplus_total, softplus_parts = safety_reserve_operating_utility_loss(
            *softplus_inputs,
            train_seed=20260801,
            global_step=0,
            total_optimizer_steps=10,
            boundary_tail_weight=0.0,
            negative_band_weight=1.0,
            glb_resource_weight=0.0,
            warmup_fraction=0.0,
            negative_band_shape="softplus",
        )
        softplus_gradient = torch.autograd.grad(
            torch.as_tensor(softplus_parts["lossEfficiency"]),
            softplus_inputs[0],
            retain_graph=False,
        )[0]
        self.assertGreater(
            float(softplus_gradient.norm()),
            100.0 * float(sigmoid_gradient.norm()),
        )
        self.assertGreater(float(sigmoid_parts["negativeBandSaturationFraction"]), 0.0)
        self.assertEqual(float(softplus_parts["negativeBandSaturationFraction"]), 0.0)

    def test_separate_efficiency_logits_isolate_base_score_gradient(self) -> None:
        inputs = list(self._inputs(requires_grad=False))
        base_logits = inputs[0].detach().clone().requires_grad_(True)
        suppression = torch.full_like(base_logits, 0.05, requires_grad=True)
        final_logits = base_logits - suppression
        efficiency_logits = base_logits.detach() - suppression
        inputs[0] = final_logits
        _total, parts = safety_reserve_operating_utility_loss(
            *inputs,
            train_seed=20260801,
            global_step=5,
            total_optimizer_steps=10,
            efficiency_logits=efficiency_logits,
            boundary_tail_weight=0.0,
            negative_band_weight=1.0,
            negative_band_shape="softplus",
            glb_resource_weight=0.0,
            warmup_fraction=0.0,
            safety_boundary_excess_weight=0.0,
        )
        efficiency = torch.as_tensor(parts["lossEfficiency"])
        base_gradient, suppression_gradient = torch.autograd.grad(
            efficiency,
            (base_logits, suppression),
            allow_unused=True,
        )
        self.assertIsNotNone(base_gradient)
        assert base_gradient is not None
        self.assertTrue(torch.equal(base_gradient, torch.zeros_like(base_gradient)))
        self.assertIsNotNone(suppression_gradient)
        assert suppression_gradient is not None
        negative = inputs[1] < 0.5
        self.assertTrue(bool((suppression_gradient[negative] < 0.0).any()))

    def test_certificate_tail_loss_separates_positives_from_only_hard_negatives(self) -> None:
        raw = torch.zeros(8, requires_grad=True)
        base = torch.tensor([-3.0, 2.0, 1.5, -2.0, -4.0, 3.0, 0.5, -1.0], requires_grad=True)
        target = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        offsets = torch.tensor([0, 4, 8], dtype=torch.long)
        weights = torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        positive_guard, hard_negative, parts = cull_certificate_tail_classification_loss(
            raw,
            base,
            target,
            offsets,
            weights,
            negative_tail_fraction=0.01,
        )
        positive_gradient = torch.autograd.grad(
            positive_guard,
            raw,
            retain_graph=True,
        )[0]
        negative_gradient = torch.autograd.grad(
            hard_negative,
            raw,
            retain_graph=True,
        )[0]
        base_gradient = torch.autograd.grad(
            positive_guard + hard_negative,
            base,
            allow_unused=True,
        )[0]
        self.assertTrue(bool((positive_gradient[target > 0.5] > 0.0).all()))
        self.assertTrue(torch.equal(positive_gradient[target <= 0.5], torch.zeros(6)))
        self.assertLess(float(negative_gradient[1]), 0.0)
        self.assertLess(float(negative_gradient[5]), 0.0)
        self.assertEqual(float(negative_gradient[3]), 0.0)
        self.assertEqual(float(negative_gradient[7]), 0.0)
        self.assertIsNone(base_gradient)
        self.assertEqual(parts["cullCertificateHardNegativeCount"], 2.0)

    def test_certificate_pair_loss_splits_positive_and_negative_gradients(self) -> None:
        logits = torch.tensor(
            [-2.0, 1.0, 0.0, -1.0, -3.0, 2.0, 1.5, -2.0],
            requires_grad=True,
        )
        target = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        offsets = torch.tensor([0, 4, 8], dtype=torch.long)
        weights = torch.tensor([10.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        positive_loss, negative_loss, parts = cull_certificate_pose_tail_pair_loss(
            logits,
            target,
            offsets,
            weights,
            positive_tail_fraction=0.01,
            negative_tail_fraction=0.01,
            pose_cvar_weight=0.0,
        )
        positive_gradient = torch.autograd.grad(
            positive_loss, logits, retain_graph=True
        )[0]
        negative_gradient = torch.autograd.grad(negative_loss, logits)[0]
        self.assertTrue(bool((positive_gradient[target > 0.5] < 0.0).all()))
        self.assertTrue(
            torch.equal(
                positive_gradient[target <= 0.5],
                torch.zeros_like(positive_gradient[target <= 0.5]),
            )
        )
        self.assertGreater(float(negative_gradient[1]), 0.0)
        self.assertGreater(float(negative_gradient[5]), 0.0)
        self.assertEqual(float(negative_gradient[2]), 0.0)
        self.assertEqual(float(negative_gradient[7]), 0.0)
        self.assertEqual(parts["cullCertificatePairPoseCount"], 2.0)
        self.assertEqual(parts["cullCertificatePairPositiveCount"], 2.0)
        self.assertEqual(parts["cullCertificatePairNegativeCount"], 2.0)

    def test_certificate_guard_can_prioritize_rare_subpose_positives(self) -> None:
        raw = torch.zeros(4, requires_grad=True)
        positive_guard, _negative, _parts = cull_certificate_tail_classification_loss(
            raw,
            torch.tensor([0.0, 0.0, 1.0, -1.0]),
            torch.tensor([1.0, 1.0, 0.0, 0.0]),
            torch.tensor([0, 4], dtype=torch.long),
            torch.tensor([1.0, 1.0, 0.0, 0.0]),
            visible_hit_rates=torch.tensor([0.04, 1.0, 0.0, 0.0]),
            positive_uniform_mix=0.0,
            rare_positive_weight=2.0,
        )
        gradient = torch.autograd.grad(positive_guard, raw)[0]
        self.assertGreater(float(gradient[0]), float(gradient[1]))

    def test_extreme_tail_separation_rewards_the_required_tail_order(self) -> None:
        target = torch.tensor([1.0, 1.0, 0.0, 0.0])
        offsets = torch.tensor([0, 4], dtype=torch.long)
        weights = torch.tensor([1.0, 0.5, 0.0, 0.0])
        bad, bad_parts = extreme_tail_separation_loss(
            torch.tensor([-1.0, 0.5, 1.0, -1.0]),
            target,
            offsets,
            weights,
            positive_tail_fraction=0.5,
            negative_tail_fraction=0.5,
        )
        good, good_parts = extreme_tail_separation_loss(
            torch.tensor([3.0, 2.0, -2.0, -3.0]),
            target,
            offsets,
            weights,
            positive_tail_fraction=0.5,
            negative_tail_fraction=0.5,
        )
        self.assertLess(float(good), float(bad))
        self.assertGreater(float(good_parts["extremeTailGap"]), 0.0)
        self.assertLess(float(bad_parts["extremeTailGap"]), 0.0)

    def test_extreme_tail_separation_pushes_selected_tails_apart(self) -> None:
        logits = torch.tensor([-1.0, 0.5, 1.0, -1.0], requires_grad=True)
        loss, parts = extreme_tail_separation_loss(
            logits,
            torch.tensor([1.0, 1.0, 0.0, 0.0]),
            torch.tensor([0, 4], dtype=torch.long),
            torch.tensor([1.0, 0.5, 0.0, 0.0]),
            positive_tail_fraction=0.5,
            negative_tail_fraction=0.5,
        )
        loss.backward()
        self.assertLess(float(logits.grad[0]), 0.0)
        self.assertGreater(float(logits.grad[2]), 0.0)
        self.assertEqual(float(logits.grad[1]), 0.0)
        self.assertEqual(float(logits.grad[3]), 0.0)
        self.assertEqual(float(parts["extremeTailPositiveCount"]), 1.0)
        self.assertEqual(float(parts["extremeTailNegativeCount"]), 1.0)

    def test_extreme_tail_membership_can_follow_a_frozen_selector(self) -> None:
        logits = torch.tensor([-1.0, 0.5, 1.0, -1.0], requires_grad=True)
        selector = torch.tensor([2.0, -3.0, -4.0, 3.0], requires_grad=True)
        loss, _parts = extreme_tail_separation_loss(
            logits,
            torch.tensor([1.0, 1.0, 0.0, 0.0]),
            torch.tensor([0, 4], dtype=torch.long),
            torch.tensor([1.0, 1.0, 0.0, 0.0]),
            positive_tail_fraction=0.5,
            negative_tail_fraction=0.5,
            selection_logits=selector,
        )
        loss.backward()

        self.assertEqual(float(logits.grad[0]), 0.0)
        self.assertLess(float(logits.grad[1]), 0.0)
        self.assertEqual(float(logits.grad[2]), 0.0)
        self.assertGreater(float(logits.grad[3]), 0.0)
        self.assertIsNone(selector.grad)

    def test_extreme_tail_current_selector_preserves_default_behavior(self) -> None:
        logits = torch.tensor([-1.0, 0.5, 1.0, -1.0])
        inputs = (
            torch.tensor([1.0, 1.0, 0.0, 0.0]),
            torch.tensor([0, 4], dtype=torch.long),
            torch.tensor([1.0, 0.5, 0.0, 0.0]),
        )
        default, _parts = extreme_tail_separation_loss(
            logits,
            *inputs,
            positive_tail_fraction=0.5,
            negative_tail_fraction=0.5,
        )
        explicit, _parts = extreme_tail_separation_loss(
            logits,
            *inputs,
            positive_tail_fraction=0.5,
            negative_tail_fraction=0.5,
            selection_logits=logits,
        )
        torch.testing.assert_close(default, explicit)

    def test_negative_only_tail_uses_positive_as_a_detached_reserve(self) -> None:
        logits = torch.tensor([-1.0, 0.5, 1.0, -1.0], requires_grad=True)
        loss, _parts = extreme_tail_separation_loss(
            logits,
            torch.tensor([1.0, 1.0, 0.0, 0.0]),
            torch.tensor([0, 4], dtype=torch.long),
            torch.tensor([1.0, 0.5, 0.0, 0.0]),
            positive_tail_fraction=0.5,
            negative_tail_fraction=0.5,
            positive_gradient_scale=0.0,
        )
        loss.backward()
        self.assertEqual(float(logits.grad[0]), 0.0)
        self.assertEqual(float(logits.grad[1]), 0.0)
        self.assertGreater(float(logits.grad[2]), 0.0)
        self.assertEqual(float(logits.grad[3]), 0.0)

    def test_safety_boundary_excess_only_pushes_selected_negatives_down(self) -> None:
        logits = torch.tensor([-1.0, 0.5, 0.2, -2.0], requires_grad=True)
        loss, parts = safety_boundary_negative_excess_loss(
            logits,
            torch.tensor([1.0, 1.0, 0.0, 0.0]),
            torch.tensor([0, 4], dtype=torch.long),
            torch.tensor([100.0, 1.0, 0.0, 0.0]),
            scope="batch",
            positive_mass_fraction=0.01,
            negative_tail_fraction=0.5,
            temperature=0.2,
        )
        loss.backward()
        self.assertEqual(float(logits.grad[0]), 0.0)
        self.assertEqual(float(logits.grad[1]), 0.0)
        self.assertGreater(float(logits.grad[2]), 0.0)
        self.assertEqual(float(logits.grad[3]), 0.0)
        self.assertEqual(float(parts["safetyBoundaryLogit"]), -1.0)
        self.assertEqual(float(parts["safetyBoundarySelectedNegativeCount"]), 1.0)

    def test_conservative_reference_boundary_only_strengthens_negative_pressure(self) -> None:
        logits = torch.tensor([-1.0, 0.5, 0.2, -2.0], requires_grad=True)
        target = torch.tensor([1.0, 1.0, 0.0, 0.0])
        weights = torch.tensor([100.0, 1.0, 0.0, 0.0])
        boundary = weighted_positive_safety_boundary_logit(
            logits,
            target,
            weights,
            positive_mass_fraction=0.01,
        )
        self.assertEqual(float(boundary), -1.0)
        self.assertFalse(boundary.requires_grad)
        loss, parts = safety_boundary_negative_excess_loss(
            logits,
            target,
            torch.tensor([0, 4], dtype=torch.long),
            weights,
            scope="batch",
            positive_mass_fraction=0.01,
            negative_tail_fraction=0.5,
            temperature=0.2,
            reference_boundary_logit=-2.0,
        )
        loss.backward()
        self.assertEqual(float(logits.grad[0]), 0.0)
        self.assertEqual(float(logits.grad[1]), 0.0)
        self.assertGreater(float(logits.grad[2]), 0.0)
        self.assertEqual(float(logits.grad[3]), 0.0)
        self.assertEqual(float(parts["safetyBoundaryBatchLogit"]), -1.0)
        self.assertEqual(float(parts["safetyBoundaryReferenceLogit"]), -2.0)
        self.assertEqual(float(parts["safetyBoundaryLogit"]), -2.0)
        with self.assertRaisesRegex(ValueError, "only valid for batch"):
            safety_boundary_negative_excess_loss(
                logits.detach(),
                target,
                torch.tensor([0, 4], dtype=torch.long),
                weights,
                scope="pose",
                reference_boundary_logit=-2.0,
            )

    def test_positive_tail_compactness_only_raises_selected_positive_tail(self) -> None:
        logits = torch.tensor([-3.0, -1.0, 0.0, 1.0, 1.0, 0.5], requires_grad=True)
        target = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
        weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
        loss, parts = weighted_positive_tail_compactness_loss(
            logits,
            target,
            weights,
            tail_mass_fraction=0.01,
            reference_quantile=0.75,
            allowed_relative_gap=0.1,
            temperature=0.2,
        )
        shifted, _ = weighted_positive_tail_compactness_loss(
            logits + 17.0,
            target,
            weights,
            tail_mass_fraction=0.01,
            reference_quantile=0.75,
            allowed_relative_gap=0.1,
            temperature=0.2,
        )
        self.assertTrue(torch.allclose(loss, shifted, atol=1e-6, rtol=1e-6))
        loss.backward()
        self.assertLess(float(logits.grad[0]), 0.0)
        self.assertEqual(float(logits.grad[1]), 0.0)
        self.assertEqual(float(logits.grad[2]), 0.0)
        self.assertEqual(float(logits.grad[3]), 0.0)
        self.assertEqual(float(logits.grad[4]), 0.0)
        self.assertEqual(float(logits.grad[5]), 0.0)
        self.assertEqual(float(parts["positiveTailSelectedCount"]), 1.0)
        self.assertGreater(float(parts["positiveTailReferenceGap"]), 0.0)

    def test_batch_safety_boundary_preserves_cross_pose_weight_mass(self) -> None:
        target = torch.tensor([1.0, 0.0, 1.0, 0.0])
        offsets = torch.tensor([0, 2, 4], dtype=torch.long)
        weights = torch.tensor([1000.0, 0.0, 1.0, 0.0])
        batch_logits = torch.tensor([-2.0, 0.0, 1.0, 0.0], requires_grad=True)
        batch_loss, batch_parts = safety_boundary_negative_excess_loss(
            batch_logits,
            target,
            offsets,
            weights,
            scope="batch",
            positive_mass_fraction=0.01,
            temperature=0.2,
        )
        batch_loss.backward()

        pose_logits = torch.tensor([-2.0, 0.0, 1.0, 0.0], requires_grad=True)
        pose_loss, pose_parts = safety_boundary_negative_excess_loss(
            pose_logits,
            target,
            offsets,
            weights,
            scope="pose",
            positive_mass_fraction=0.01,
            temperature=0.2,
            pose_cvar_weight=0.0,
        )
        pose_loss.backward()
        self.assertEqual(float(batch_parts["safetyBoundaryLogit"]), -2.0)
        self.assertEqual(float(pose_parts["safetyBoundaryLogit"]), -0.5)
        self.assertGreater(float(batch_logits.grad[3]), float(pose_logits.grad[3]))

    def test_extreme_tail_separation_is_translation_invariant(self) -> None:
        logits = torch.tensor([3.0, 2.0, -2.0, -3.0])
        inputs = (
            torch.tensor([1.0, 1.0, 0.0, 0.0]),
            torch.tensor([0, 4], dtype=torch.long),
            torch.tensor([1.0, 0.5, 0.0, 0.0]),
        )
        original, _parts = extreme_tail_separation_loss(
            logits,
            *inputs,
            positive_tail_fraction=0.5,
            negative_tail_fraction=0.5,
        )
        shifted, _parts = extreme_tail_separation_loss(
            logits + 17.0,
            *inputs,
            positive_tail_fraction=0.5,
            negative_tail_fraction=0.5,
        )
        self.assertTrue(torch.allclose(original, shifted, atol=1e-6, rtol=1e-6))

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

    def test_projection_ignores_safety_only_projection_head_coordinates(self) -> None:
        safety = [torch.tensor([1.0, 0.0]), torch.tensor([100.0])]
        relation = [torch.tensor([-1.0, 1.0]), None]
        zeros = [torch.zeros(2), None]

        projected, statistics = project_operating_utility_gradient_groups(
            safety,
            relation,
            zeros,
            zeros,
            relation_norm_cap=10.0,
            schedule_norm_cap=10.0,
            efficiency_norm_cap=10.0,
        )

        self.assertEqual(
            statistics["relationGradientProjectionFallbackZeroed"], 0.0
        )
        self.assertEqual(statistics["relationGradientSharedSafetyNorm"], 1.0)
        self.assertGreater(float(projected["relation"][0][1]), 0.0)
        self.assertGreaterEqual(
            float(torch.dot(safety[0], projected["relation"][0])),
            -1e-7,
        )

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

    def test_staged_rvl_budget_and_tail_are_reported_in_total(self) -> None:
        inputs = self._inputs(requires_grad=True)
        total, parts = safety_reserve_operating_utility_loss(
            *inputs,
            train_seed=20260817,
            global_step=5,
            total_optimizer_steps=100,
            boundary_tail_weight=0.0,
            negative_band_weight=0.0,
            glb_resource_weight=0.0,
            tail_separation_weight=0.6,
            tail_positive_fraction=0.5,
            tail_negative_fraction=0.5,
            tail_ramp_fraction=0.1,
            rvl_budget_initial_scale=0.1,
            rvl_budget_start_fraction=0.1,
            rvl_budget_ramp_fraction=0.25,
        )
        expected = (
            torch.as_tensor(parts["lossSafetyRvl"])
            + torch.as_tensor(parts["lossExtremeTailSeparationScaled"])
        )
        self.assertTrue(torch.allclose(total, expected, atol=1e-7, rtol=1e-7))
        self.assertEqual(float(parts["rvlBudgetScheduleScale"]), 0.1)
        self.assertGreater(float(parts["extremeTailRampScale"]), 0.0)
        total.backward()
        self.assertTrue(bool(torch.isfinite(inputs[0].grad).all()))

    def test_weighted_and_uniform_positive_tails_are_both_in_total(self) -> None:
        inputs = self._inputs(requires_grad=True)
        total, parts = safety_reserve_operating_utility_loss(
            *inputs,
            train_seed=20260818,
            global_step=50,
            total_optimizer_steps=100,
            boundary_tail_weight=0.0,
            negative_band_weight=0.0,
            glb_resource_weight=0.0,
            tail_separation_weight=0.4,
            tail_positive_fraction=0.25,
            tail_positive_importance_mix=1.0,
            coverage_tail_separation_weight=0.3,
            coverage_tail_positive_fraction=0.5,
            tail_negative_fraction=0.5,
            tail_ramp_fraction=0.0,
        )

        importance = torch.as_tensor(parts["lossImportanceTailSeparationScaled"])
        coverage = torch.as_tensor(
            parts["lossUniformCoverageTailSeparationScaled"]
        )
        combined = torch.as_tensor(parts["lossExtremeTailSeparationScaled"])
        self.assertGreater(float(importance), 0.0)
        self.assertGreater(float(coverage), 0.0)
        self.assertTrue(torch.allclose(combined, importance + coverage))
        self.assertTrue(
            torch.allclose(
                total,
                torch.as_tensor(parts["lossSafetyRvl"]) + combined,
                atol=1e-7,
                rtol=1e-7,
            )
        )
        self.assertGreater(
            float(parts["coverageTailExtremeTailPositiveCount"]), 0.0
        )
        total.backward()
        self.assertTrue(bool(torch.isfinite(inputs[0].grad).all()))

    def test_zero_tail_weight_preserves_the_registered_loss_path(self) -> None:
        inputs = self._inputs(requires_grad=True)
        with mock.patch(
            "common.safety_reserve_operating_utility_loss.extreme_tail_separation_loss",
            side_effect=AssertionError("disabled tail path must not execute"),
        ):
            total, parts = safety_reserve_operating_utility_loss(
                *inputs,
                train_seed=20260801,
                global_step=50,
                total_optimizer_steps=100,
                tail_separation_weight=0.0,
            )
        expected = (
            torch.as_tensor(parts["lossSafetyRvl"])
            + torch.as_tensor(parts["lossBoundaryTailScaled"])
            + torch.as_tensor(parts["lossEfficiency"])
        )
        self.assertTrue(torch.allclose(total, expected, atol=1e-7, rtol=1e-7))
        self.assertEqual(float(parts["lossExtremeTailSeparationScaled"]), 0.0)

    def test_tail_can_be_projected_as_an_efficiency_objective(self) -> None:
        inputs = self._inputs(requires_grad=True)
        total, parts = safety_reserve_operating_utility_loss(
            *inputs,
            train_seed=20260817,
            global_step=50,
            total_optimizer_steps=100,
            negative_band_weight=0.0,
            glb_resource_weight=0.0,
            tail_separation_weight=0.5,
            tail_positive_fraction=0.5,
            tail_negative_fraction=0.5,
            tail_positive_gradient_scale=0.0,
            tail_objective_group="efficiency",
        )
        self.assertEqual(float(parts["lossExtremeTailSeparationSafetyScaled"]), 0.0)
        self.assertGreater(float(parts["lossExtremeTailSeparationEfficiencyScaled"]), 0.0)
        expected = torch.as_tensor(parts["lossSafety"]) + torch.as_tensor(parts["lossEfficiency"])
        self.assertTrue(torch.allclose(total, expected, atol=1e-7, rtol=1e-7))
        safety_gradient = torch.autograd.grad(
            torch.as_tensor(parts["lossSafety"]), inputs[0], retain_graph=True
        )[0]
        efficiency_gradient = torch.autograd.grad(
            torch.as_tensor(parts["lossEfficiency"]), inputs[0], retain_graph=False
        )[0]
        projected, _statistics = project_operating_utility_gradient_groups(
            [safety_gradient],
            [torch.zeros_like(safety_gradient)],
            [torch.zeros_like(safety_gradient)],
            [efficiency_gradient],
            efficiency_norm_cap=0.25,
        )
        self.assertGreaterEqual(
            float(torch.dot(safety_gradient, projected["efficiency"][0])),
            -1e-7,
        )

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
