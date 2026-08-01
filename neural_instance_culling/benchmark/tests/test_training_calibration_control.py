from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

MODEL_DIR = Path(__file__).resolve().parents[2] / "model"
sys.path.insert(0, str(MODEL_DIR))

from train_directional_occlusion_proxy_encoder import (  # noqa: E402
    resolve_loss_profile,
    select_diagnostic_calibration_workpoint,
)
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


if __name__ == "__main__":
    unittest.main()
