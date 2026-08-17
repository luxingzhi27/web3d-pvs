from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from neural_instance_culling.model.train_bounded_relation_survival_moment_safety import (
    CALIBRATION_FLOOR,
    _calibration_workpoints,
    _evaluation_batch_limits,
    _instance_calibration_blend,
    _instance_calibration_reliability,
    _operating_threshold_metrics,
    _resolve_split,
    _save_safe_checkpoint_alias,
    _v4_objective_groups,
)
from neural_instance_culling.model.current_pvs_utils import score_distribution_summary


class BoundedRelationSurvivalMomentSafetyTrainingTest(unittest.TestCase):
    def test_score_distribution_reports_extreme_safety_tails(self) -> None:
        summary = score_distribution_summary(
            np.asarray([0.01, 0.60, 0.90, 0.20, 0.80, 0.99]),
            np.asarray([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
            np.asarray([10.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        )
        for key in (
            "positiveWeightedQ005",
            "positiveWeightedQ01",
            "negativeQ99",
            "negativeQ995",
            "positiveNegativeGapQ01Q99",
            "positiveNegativeGapQ005Q995",
            "rocAuc",
            "averagePrecision",
            "weightedRocAuc",
        ):
            self.assertIn(key, summary)
            self.assertIsNotNone(summary[key])
        self.assertLess(summary["positiveNegativeGapQ005Q995"], 0.0)
        self.assertAlmostEqual(summary["rocAuc"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["averagePrecision"], 0.5)
        self.assertAlmostEqual(summary["weightedRocAuc"], 1.0 / 12.0)

    def test_test_split_is_rejected_without_reading_it(self) -> None:
        dataset = SimpleNamespace(split_ids={"train", "calibration", "validation", "test"})
        with self.assertRaisesRegex(ValueError, "test is forbidden"):
            _resolve_split(dataset, "test", "validation")

    def test_validation_values_do_not_enter_calibration_workpoint_selection(self) -> None:
        calibration_rows = [
            {
                "threshold": 0.02,
                "aggregateWeightedRecall": 0.995,
                "aggregateWeightedRecallLowerConfidenceBound": 0.992,
                "agg_useful_cull": 0.20,
                "agg_balanced_accuracy": 0.80,
                "agg_precision": 0.80,
                "avg_pred_count": 10.0,
            },
            {
                "threshold": 0.05,
                "aggregateWeightedRecall": 0.996,
                "aggregateWeightedRecallLowerConfidenceBound": 0.993,
                "agg_useful_cull": 0.40,
                "agg_balanced_accuracy": 0.85,
                "agg_precision": 0.85,
                "avg_pred_count": 8.0,
            },
        ]
        selected, diagnostic, frozen = _calibration_workpoints(calibration_rows)
        self.assertIsNotNone(selected)
        self.assertIsNotNone(diagnostic)
        self.assertIs(frozen, selected)
        self.assertEqual(frozen["threshold"], 0.05)
        self.assertEqual(CALIBRATION_FLOOR, 0.99)

    def test_four_objective_groups_sum_to_logged_total(self) -> None:
        values = [torch.tensor(value, requires_grad=True) for value in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)]
        groups = _v4_objective_groups(
            *values,
            survival_weight=0.25,
            relation_consistency_weight=0.10,
            utility_weight=0.10,
            download_weight=0.10,
            regularization_weight=1e-5,
            instance_calibration_regularization_weight=0.02,
        )
        summed = groups["safety"] + groups["relation"] + groups["schedule"] + groups["efficiency"]
        self.assertTrue(torch.allclose(groups["total"], summed))
        groups["total"].backward()
        self.assertTrue(all(value.grad is not None for value in values))

    def test_instance_calibration_schedule_warms_up_then_reaches_one(self) -> None:
        values = [
            _instance_calibration_blend(
                step,
                100,
                warmup_fraction=0.10,
                ramp_fraction=0.20,
            )
            for step in (0, 9, 10, 19, 29, 99)
        ]
        self.assertEqual(values[:2], [0.0, 0.0])
        self.assertGreater(values[2], 0.0)
        self.assertTrue(all(left <= right for left, right in zip(values, values[1:])))
        self.assertEqual(values[-1], 1.0)

    def test_instance_calibration_reliability_reads_train_references_only(self) -> None:
        class Dataset:
            @staticmethod
            def frustum_slice(pose_index: int) -> np.ndarray:
                return {
                    0: np.asarray([0, 1], dtype=np.uint32),
                    1: np.asarray([0, 2], dtype=np.uint32),
                }[pose_index]

        reliability, meta = _instance_calibration_reliability(
            Dataset(),
            SimpleNamespace(pose_indices=np.asarray([0, 1], dtype=np.int64)),
            {"instance": np.asarray([0, 0, 2], dtype=np.uint32)},
            4,
        )
        self.assertEqual(reliability.shape, (4,))
        self.assertEqual(reliability[3], 0.0)
        self.assertGreater(reliability[0], reliability[1])
        self.assertEqual(meta["sourceSplit"], "train")
        self.assertEqual(len(meta["sha256Float32"]), 64)

    def test_logged_operating_thresholds_are_seed_step_deterministic(self) -> None:
        first, first_metrics = _operating_threshold_metrics(20260801, 3)
        repeated, repeated_metrics = _operating_threshold_metrics(20260801, 3)
        changed, _changed_metrics = _operating_threshold_metrics(20260801, 4)
        self.assertEqual(first, repeated)
        self.assertEqual(first_metrics, repeated_metrics)
        self.assertNotEqual(first, changed)
        self.assertEqual(first_metrics["operatingThresholdCount"], float(len(first)))
        self.assertTrue(all(np.isfinite(value) for value in first))

    def test_diagnostic_pose_limit_is_exact(self) -> None:
        self.assertEqual(_evaluation_batch_limits(0, 4), (4, None))
        self.assertEqual(_evaluation_batch_limits(2, 4), (2, 1))
        self.assertEqual(_evaluation_batch_limits(5, 4), (1, 5))
        with self.assertRaises(ValueError):
            _evaluation_batch_limits(-1, 4)

    def test_safe_checkpoint_alias_is_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _save_safe_checkpoint_alias(
                {"schema": "unit-test", "tensor": torch.tensor([1.0, 2.0])}, root
            )
            self.assertEqual(
                (root / "best_safe.pt").read_bytes(),
                (root / "best.pt").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
