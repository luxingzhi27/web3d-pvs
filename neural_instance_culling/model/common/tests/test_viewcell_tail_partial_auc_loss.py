from __future__ import annotations

import unittest

import torch

from neural_instance_culling.model.common.viewcell_tail_partial_auc_loss import (
    viewcell_tail_partial_auc_loss,
)


class ViewcellTailPartialAucLossTests(unittest.TestCase):
    @staticmethod
    def _inputs(*, residual: torch.Tensor | None = None) -> tuple[torch.Tensor, ...]:
        if residual is None:
            residual = torch.zeros(8, dtype=torch.float32)
        base = torch.tensor([-3.0, 1.0, 2.0, -1.0, 0.0, 3.0, 1.5, -2.0])
        target = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        hit_rates = torch.tensor([0.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0])
        weights = torch.tensor([1.0, 10.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0])
        offsets = torch.tensor([0, 4, 8], dtype=torch.long)
        return residual, base, target, hit_rates, weights, offsets

    @staticmethod
    def _pair_kwargs() -> dict[str, float | int]:
        return {
            "positive_tail_mass_fraction": 1.0,
            "positive_tail_count_cap": 8,
            "min_positive_count": 1,
            "negative_top_fraction": 1.0,
            "min_negative_count": 1,
            "max_negative_count": 8,
            "pose_cvar_fraction": 1.0,
            "pose_cvar_weight": 0.0,
            "negative_positive_residual_weight": 0.0,
            "selected_positive_negative_residual_weight": 0.0,
            "residual_l2_weight": 0.0,
        }

    def test_pair_gradient_has_safety_boundary_direction(self) -> None:
        residual = torch.zeros(4, requires_grad=True)
        base = torch.tensor([-1.0, 1.0, 0.5, -0.5])
        target = torch.tensor([1.0, 1.0, 0.0, 0.0])
        hit = torch.tensor([0.0, 1.0, 1.0, 1.0])
        weights = torch.tensor([1.0, 1.0, 0.0, 0.0])
        loss, _ = viewcell_tail_partial_auc_loss(
            residual,
            base,
            target,
            hit,
            weights,
            torch.tensor([0, 4]),
            **self._pair_kwargs(),
        )
        loss.backward()
        self.assertLess(float(residual.grad[0]), 0.0)
        self.assertGreater(float(residual.grad[2]), 0.0)
        self.assertLess(float(residual.grad[1]), 0.0)
        self.assertGreater(float(residual.grad[3]), 0.0)
        self.assertGreater(abs(float(residual.grad[0])), abs(float(residual.grad[1])))

    def test_tail_selection_respects_mass_cap_and_negative_min_max(self) -> None:
        residual, base, target, hit, weights, offsets = self._inputs()
        _, stats = viewcell_tail_partial_auc_loss(
            residual,
            base,
            target,
            hit,
            weights,
            offsets,
            positive_tail_mass_fraction=0.20,
            positive_tail_count_cap=1,
            min_positive_count=1,
            negative_top_fraction=0.01,
            min_negative_count=2,
            max_negative_count=2,
            pose_cvar_weight=0.0,
        )
        self.assertEqual(stats["selectedPositiveCount"], 2.0)
        self.assertEqual(stats["selectedNegativeCount"], 4.0)
        self.assertEqual(stats["poseCount"], 2.0)

    def test_current_score_can_refresh_tail_membership(self) -> None:
        residual = torch.tensor([5.0, -5.0, -5.0, 5.0])
        base = torch.tensor([-3.0, 1.0, 2.0, 0.0])
        target = torch.tensor([1.0, 1.0, 0.0, 0.0])
        hit = torch.ones(4)
        weights = torch.tensor([1.0, 1.0, 0.0, 0.0])
        kwargs = {
            **self._pair_kwargs(),
            "positive_tail_mass_fraction": 0.01,
            "positive_tail_count_cap": 1,
            "negative_top_fraction": 0.01,
            "max_negative_count": 1,
        }
        _, frozen_stats = viewcell_tail_partial_auc_loss(
            residual,
            base,
            target,
            hit,
            weights,
            torch.tensor([0, 4]),
            **kwargs,
        )
        _, current_stats = viewcell_tail_partial_auc_loss(
            residual,
            base,
            target,
            hit,
            weights,
            torch.tensor([0, 4]),
            tail_selection_logits=base + residual,
            **kwargs,
        )
        self.assertEqual(float(frozen_stats["selectedPositiveResidualMean"]), 5.0)
        self.assertEqual(float(current_stats["selectedPositiveResidualMean"]), -5.0)
        self.assertEqual(float(frozen_stats["selectedNegativeResidualMean"]), -5.0)
        self.assertEqual(float(current_stats["selectedNegativeResidualMean"]), 5.0)

    def test_pose_cvar_is_the_upper_tail_of_pose_losses(self) -> None:
        residual = torch.zeros(8)
        base = torch.tensor([-4.0, 2.0, 1.0, 0.0, -0.2, 0.3, 0.1, -0.1])
        target = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        hit = torch.ones(8)
        weights = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        kwargs = self._pair_kwargs()
        kwargs.update(
            {
                "positive_tail_mass_fraction": 1.0,
                "negative_top_fraction": 1.0,
                "pose_cvar_fraction": 0.5,
                "pose_cvar_weight": 1.0,
            }
        )
        loss, stats = viewcell_tail_partial_auc_loss(
            residual,
            base,
            target,
            hit,
            weights,
            torch.tensor([0, 4, 8]),
            **kwargs,
        )
        pose_mean = float(stats["lossPoseMean"])
        pose_cvar = float(stats["lossPoseCvar"])
        self.assertGreaterEqual(pose_cvar, pose_mean)
        self.assertAlmostEqual(float(loss), pose_mean + pose_cvar, places=6)

    def test_power_importance_preserves_visual_utility_ratio(self) -> None:
        base = torch.zeros(3)
        target = torch.tensor([1.0, 1.0, 0.0])
        hit = torch.ones(3)
        weights = torch.tensor([1.0, 9.0, 0.0])

        log_residual = torch.zeros(3, requires_grad=True)
        log_loss, _ = viewcell_tail_partial_auc_loss(
            log_residual,
            base,
            target,
            hit,
            weights,
            torch.tensor([0, 3]),
            positive_importance_transform="log1p",
            **self._pair_kwargs(),
        )
        log_loss.backward()

        linear_residual = torch.zeros(3, requires_grad=True)
        linear_loss, stats = viewcell_tail_partial_auc_loss(
            linear_residual,
            base,
            target,
            hit,
            weights,
            torch.tensor([0, 3]),
            positive_importance_transform="power",
            positive_importance_power=1.0,
            **self._pair_kwargs(),
        )
        linear_loss.backward()

        log_ratio = abs(float(log_residual.grad[1] / log_residual.grad[0]))
        linear_ratio = abs(
            float(linear_residual.grad[1] / linear_residual.grad[0])
        )
        self.assertLess(log_ratio, linear_ratio)
        self.assertAlmostEqual(linear_ratio, 9.0, places=5)
        self.assertAlmostEqual(
            float(stats["selectedPositiveWeightMassFraction"]),
            1.0,
        )
        self.assertEqual(float(stats["positiveImportancePower"]), 1.0)

    def test_cross_pose_pair_targets_the_shared_global_boundary(self) -> None:
        residual = torch.zeros(4, requires_grad=True)
        base = torch.tensor([-2.0, -3.0, 3.0, 2.0])
        target = torch.tensor([1.0, 0.0, 1.0, 0.0])
        hit = torch.ones(4)
        weights = torch.tensor([1.0, 0.0, 1.0, 0.0])
        kwargs = {
            **self._pair_kwargs(),
            "margin": 0.0,
            "temperature": 0.25,
        }
        local_loss, _ = viewcell_tail_partial_auc_loss(
            residual,
            base,
            target,
            hit,
            weights,
            torch.tensor([0, 2, 4]),
            cross_pose_pair_weight=0.0,
            **kwargs,
        )
        global_loss, stats = viewcell_tail_partial_auc_loss(
            residual,
            base,
            target,
            hit,
            weights,
            torch.tensor([0, 2, 4]),
            cross_pose_pair_weight=1.0,
            **kwargs,
        )
        self.assertGreater(float(global_loss), float(local_loss))
        self.assertGreater(float(stats["lossCrossPosePair"]), 0.0)
        self.assertGreater(float(stats["crossPoseViolationFraction"]), 0.0)
        global_loss.backward()
        self.assertLess(float(residual.grad[0]), 0.0)
        self.assertGreater(float(residual.grad[3]), 0.0)

    def test_cross_pose_pair_excludes_same_pose_pairs(self) -> None:
        _, stats = viewcell_tail_partial_auc_loss(
            torch.zeros(4),
            torch.tensor([-2.0, -3.0, 3.0, 2.0]),
            torch.tensor([1.0, 0.0, 1.0, 0.0]),
            torch.ones(4),
            torch.tensor([1.0, 0.0, 1.0, 0.0]),
            torch.tensor([0, 4]),
            cross_pose_pair_weight=1.0,
            **self._pair_kwargs(),
        )
        self.assertEqual(float(stats["lossCrossPosePair"]), 0.0)
        self.assertEqual(float(stats["crossPoseViolationFraction"]), 0.0)

    def test_global_tail_pair_uses_the_batch_extremes(self) -> None:
        residual = torch.zeros(4, requires_grad=True)
        loss, stats = viewcell_tail_partial_auc_loss(
            residual,
            torch.tensor([-2.0, -3.0, 3.0, 2.0]),
            torch.tensor([1.0, 0.0, 1.0, 0.0]),
            torch.ones(4),
            torch.tensor([1.0, 0.0, 1.0, 0.0]),
            torch.tensor([0, 2, 4]),
            global_tail_pair_weight=1.0,
            margin=0.0,
            temperature=0.25,
            **self._pair_kwargs(),
        )
        self.assertGreater(float(stats["lossGlobalTailPair"]), 0.0)
        self.assertGreater(float(stats["globalTailViolationFraction"]), 0.0)
        self.assertEqual(float(stats["globalTailPositiveCount"]), 2.0)
        self.assertEqual(float(stats["globalTailNegativeCount"]), 2.0)
        loss.backward()
        self.assertLess(float(residual.grad[0]), 0.0)
        self.assertGreater(float(residual.grad[3]), 0.0)

    def test_multi_pose_mean_and_zero_mean_diagnostics(self) -> None:
        residual = torch.tensor([-1.0, 1.0, 2.0, -2.0])
        loss, stats = viewcell_tail_partial_auc_loss(
            residual,
            torch.tensor([-1.0, 1.0, 0.5, -0.5]),
            torch.tensor([1.0, 0.0, 1.0, 0.0]),
            torch.ones(4),
            torch.tensor([1.0, 0.0, 2.0, 0.0]),
            torch.tensor([0, 2, 4]),
            **self._pair_kwargs(),
        )
        self.assertEqual(stats["poseCount"], 2.0)
        self.assertAlmostEqual(float(stats["residualMean"]), 0.0)
        self.assertAlmostEqual(float(stats["poseResidualMeanAbs"]), 1.5)
        self.assertTrue(bool(torch.isfinite(loss)))

    def test_one_class_poses_remain_finite_and_regularized(self) -> None:
        residual = torch.tensor([0.4, -0.5, 0.3], requires_grad=True)
        loss, stats = viewcell_tail_partial_auc_loss(
            residual,
            torch.tensor([0.0, 1.0, -1.0]),
            torch.tensor([0.0, 0.0, 1.0]),
            torch.ones(3),
            torch.tensor([0.0, 0.0, 1.0]),
            torch.tensor([0, 1, 2, 3]),
            pose_cvar_weight=0.0,
        )
        self.assertEqual(stats["poseCount"], 0.0)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertGreater(float(loss), 0.0)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(residual.grad).all()))

    def test_all_positive_guard_penalizes_negative_positive_residuals(self) -> None:
        inputs = self._inputs(
            residual=torch.tensor(
                [-0.5, 0.2, 0.0, 0.0, -0.3, 0.0, 0.0, 0.0],
                requires_grad=True,
            )
        )
        base_loss, _ = viewcell_tail_partial_auc_loss(
            *inputs,
            **self._pair_kwargs(),
        )
        guarded_loss, stats = viewcell_tail_partial_auc_loss(
            *inputs,
            all_positive_negative_residual_weight=2.0,
            **self._pair_kwargs(),
        )
        self.assertGreater(float(guarded_loss), float(base_loss))
        self.assertGreater(float(stats["lossAllPositiveNegativeResidual"]), 0.0)
        guarded_loss.backward()
        self.assertLess(float(inputs[0].grad[0]), 0.0)
        self.assertLess(float(inputs[0].grad[4]), 0.0)

    def test_tail_classification_directly_signs_selected_corrections(self) -> None:
        base = torch.zeros(4)
        target = torch.tensor([1.0, 1.0, 0.0, 0.0])
        hit = torch.ones(4)
        weights = torch.tensor([1.0, 2.0, 0.0, 0.0])
        zero_residual = torch.zeros(4, requires_grad=True)
        zero_loss, zero_stats = viewcell_tail_partial_auc_loss(
            zero_residual,
            base,
            target,
            hit,
            weights,
            torch.tensor([0, 4]),
            tail_classification_weight=1.0,
            tail_classification_margin=0.2,
            **self._pair_kwargs(),
        )
        zero_loss.backward()
        self.assertTrue(bool((zero_residual.grad[:2] < 0.0).all()))
        self.assertTrue(bool((zero_residual.grad[2:] > 0.0).all()))

        _, corrected_stats = viewcell_tail_partial_auc_loss(
            torch.tensor([0.5, 0.5, -0.5, -0.5]),
            base,
            target,
            hit,
            weights,
            torch.tensor([0, 4]),
            tail_classification_weight=1.0,
            tail_classification_margin=0.2,
            **self._pair_kwargs(),
        )
        self.assertLess(
            float(corrected_stats["lossTailClassification"]),
            float(zero_stats["lossTailClassification"]),
        )
        self.assertEqual(
            float(corrected_stats["tailPositiveCorrectionSatisfiedFraction"]),
            1.0,
        )
        self.assertEqual(
            float(corrected_stats["tailNegativeCorrectionSatisfiedFraction"]),
            1.0,
        )

    def test_invalid_shapes_values_offsets_and_parameters_are_rejected(self) -> None:
        inputs = self._inputs()
        with self.assertRaisesRegex(ValueError, "align"):
            viewcell_tail_partial_auc_loss(inputs[0], inputs[1][:-1], *inputs[2:])
        with self.assertRaisesRegex(ValueError, "finite"):
            bad = inputs[0].clone()
            bad[0] = float("nan")
            viewcell_tail_partial_auc_loss(bad, *inputs[1:])
        with self.assertRaisesRegex(ValueError, "binary"):
            bad_target = inputs[2].clone()
            bad_target[0] = 0.5
            viewcell_tail_partial_auc_loss(inputs[0], inputs[1], bad_target, *inputs[3:])
        with self.assertRaisesRegex(ValueError, "pose_offsets"):
            viewcell_tail_partial_auc_loss(
                *inputs[:-1], torch.tensor([1, 8], dtype=torch.long)
            )
        with self.assertRaisesRegex(ValueError, "positive_tail_mass_fraction"):
            viewcell_tail_partial_auc_loss(
                *inputs, positive_tail_mass_fraction=0.0
            )
        with self.assertRaisesRegex(ValueError, "max_negative_count"):
            viewcell_tail_partial_auc_loss(*inputs, max_negative_count=0)
        with self.assertRaisesRegex(ValueError, "temperature"):
            viewcell_tail_partial_auc_loss(*inputs, temperature=0.0)
        with self.assertRaisesRegex(ValueError, "tail_classification_margin"):
            viewcell_tail_partial_auc_loss(
                *inputs, tail_classification_margin=-0.1
            )
        with self.assertRaisesRegex(ValueError, "align"):
            viewcell_tail_partial_auc_loss(
                *inputs,
                tail_selection_logits=torch.zeros(2),
            )
        with self.assertRaisesRegex(ValueError, "positive_importance_transform"):
            viewcell_tail_partial_auc_loss(
                *inputs,
                positive_importance_transform="invalid",
            )
        with self.assertRaisesRegex(ValueError, "positive_importance_power"):
            viewcell_tail_partial_auc_loss(
                *inputs,
                positive_importance_transform="power",
                positive_importance_power=0.0,
            )
        with self.assertRaisesRegex(ValueError, "cross_pose_pair_weight"):
            viewcell_tail_partial_auc_loss(
                *inputs,
                cross_pose_pair_weight=-0.1,
            )
        with self.assertRaisesRegex(ValueError, "global_tail_pair_weight"):
            viewcell_tail_partial_auc_loss(
                *inputs,
                global_tail_pair_weight=-0.1,
            )


if __name__ == "__main__":
    unittest.main()
