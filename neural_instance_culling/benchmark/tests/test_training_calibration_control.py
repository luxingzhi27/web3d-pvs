from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

MODEL_DIR = Path(__file__).resolve().parents[2] / "model"
sys.path.insert(0, str(MODEL_DIR))

from train_directional_occlusion_proxy_encoder import (  # noqa: E402
    relative_checkpoint_selection_key,
    resolve_loss_profile,
    select_diagnostic_calibration_workpoint,
)
from common.threshold_selection import select_weighted_precision_workpoint  # noqa: E402
from current_pvs_utils import threshold_grid  # noqa: E402


class TrainingCalibrationControlTests(unittest.TestCase):
    def test_diagnostic_row_prefers_recall_but_is_not_a_safe_selection(self) -> None:
        rows = [
            {
                "threshold": 0.2,
                "pose_weighted_recall": 0.988,
                "pose_precision": 0.70,
                "pose_f1": 0.72,
                "avg_pred_count": 100.0,
            },
            {
                "threshold": 0.1,
                "pose_weighted_recall": 0.987,
                "pose_precision": 0.95,
                "pose_f1": 0.80,
                "avg_pred_count": 120.0,
            },
        ]
        selected = select_diagnostic_calibration_workpoint(rows)
        self.assertEqual(selected["threshold"], 0.2)
        self.assertLess(selected["pose_weighted_recall"], 0.99)

    def test_empty_calibration_rows_fail_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty list"):
            select_diagnostic_calibration_workpoint([])

    def test_safe_checkpoint_always_ranks_above_unsafe_checkpoint(self) -> None:
        unsafe = relative_checkpoint_selection_key(
            {
                "pose_weighted_recall": 0.989,
                "pose_balanced_accuracy": 0.99,
                "pose_precision": 0.99,
                "avg_pred_count": 10.0,
            },
            0.99,
            calibration_safe=True,
        )
        safe = relative_checkpoint_selection_key(
            {
                "pose_weighted_recall": 0.991,
                "pose_f1": 0.50,
                "pose_precision": 0.40,
                "avg_pred_count": 100.0,
            },
            0.99,
            calibration_safe=True,
        )
        self.assertGreater(safe, unsafe)

    def test_unsafe_pilot_still_has_a_relative_checkpoint_ranking(self) -> None:
        lower_recall = relative_checkpoint_selection_key(
            {
                "pose_weighted_recall": 0.980,
                "pose_balanced_accuracy": 0.90,
                "pose_precision": 0.80,
                "avg_pred_count": 100.0,
            },
            0.99,
            calibration_safe=True,
        )
        higher_recall = relative_checkpoint_selection_key(
            {
                "pose_weighted_recall": 0.985,
                "pose_balanced_accuracy": 0.70,
                "pose_precision": 0.60,
                "avg_pred_count": 200.0,
            },
            0.99,
            calibration_safe=True,
        )
        self.assertGreater(higher_recall, lower_recall)

    def test_threshold_grid_covers_zero_and_low_score_tail(self) -> None:
        thresholds = threshold_grid()
        self.assertEqual(float(thresholds[0]), 0.0)
        self.assertLessEqual(float(thresholds[1]), 1e-8 * 1.01)
        self.assertLessEqual(float(thresholds[-1]), 0.9)
        self.assertEqual(len(thresholds), len(set(float(value) for value in thresholds)))

    def test_explicit_rvl_off_overrides_profile_default(self) -> None:
        args = SimpleNamespace(loss_profile="rvl_strong_v2", rvl_mode="off")
        resolve_loss_profile(args)
        self.assertEqual(args.rvl_mode, "off")

    def test_profile_rvl_mode_is_used_when_cli_does_not_override(self) -> None:
        args = SimpleNamespace(loss_profile="rvl_strong_v2", rvl_mode=None)
        resolve_loss_profile(args)
        self.assertEqual(args.rvl_mode, "evidence")

    def test_safety_workpoint_can_require_ordinary_pose_recall(self) -> None:
        rows = [
            {
                "threshold": 0.2,
                "pose_recall": 0.94,
                "pose_weighted_recall": 0.997,
                "pose_precision": 0.80,
                "pose_f1": 0.85,
                "avg_pred_count": 30.0,
            },
            {
                "threshold": 0.1,
                "pose_recall": 0.95,
                "pose_weighted_recall": 0.993,
                "pose_precision": 0.60,
                "pose_f1": 0.70,
                "avg_pred_count": 50.0,
            },
        ]
        selected = select_weighted_precision_workpoint(
            rows,
            target_weighted_recall=0.99,
            minimum_point_estimate=0.9925,
            minimum_pose_recall=0.95,
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected["threshold"], 0.1)


if __name__ == "__main__":
    unittest.main()
