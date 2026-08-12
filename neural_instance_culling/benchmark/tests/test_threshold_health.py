from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "model"
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

from common.threshold_selection import (  # noqa: E402
    select_healthy_weighted_cull_workpoint,
    summarize_threshold_health,
)


class ThresholdHealthTest(unittest.TestCase):
    def test_low_only_safe_workpoints_are_flagged_unhealthy(self) -> None:
        rows = [
            {"threshold": 0.01, "pose_weighted_recall": 0.999, "weighted_recall_lower_confidence_bound": 0.995, "scoreDistribution": {"positiveNegativeGapQ05Q95": 0.2}},
            {"threshold": 0.10, "pose_weighted_recall": 0.998, "weighted_recall_lower_confidence_bound": 0.994},
        ]
        health = summarize_threshold_health(rows, 0.99, 0.99)
        self.assertFalse(health["hasHealthySafeWorkpoint"])
        self.assertTrue(health["safeOnlyBelowHealthyInterval"])
        self.assertEqual(health["interpretation"], "safe_but_score_distribution_unhealthy")

    def test_middle_safe_workpoint_is_healthy(self) -> None:
        rows = [
            {"threshold": 0.40, "pose_weighted_recall": 0.998, "weighted_recall_lower_confidence_bound": 0.994, "scoreDistribution": {"positiveNegativeGapQ05Q95": 0.2}},
            {"threshold": 0.50, "pose_weighted_recall": 0.997, "weighted_recall_lower_confidence_bound": 0.993},
        ]
        health = summarize_threshold_health(rows, 0.99, 0.99)
        self.assertTrue(health["hasHealthySafeWorkpoint"])
        self.assertEqual(health["interpretation"], "healthy_anchor_and_middle_safe_workpoint")
        self.assertTrue(health["anchorSafe"])

    def test_middle_threshold_without_safe_anchor_is_not_healthy(self) -> None:
        rows = [
            {"threshold": 0.40, "pose_weighted_recall": 0.998, "weighted_recall_lower_confidence_bound": 0.994, "scoreDistribution": {"positiveNegativeGapQ05Q95": 0.2}},
            {"threshold": 0.50, "pose_weighted_recall": 0.989, "weighted_recall_lower_confidence_bound": 0.980},
        ]
        health = summarize_threshold_health(rows, 0.99, 0.99)
        self.assertFalse(health["hasHealthySafeWorkpoint"])
        self.assertTrue(health["middleSafeButAnchorUnsafe"])
        self.assertEqual(health["interpretation"], "middle_safe_but_anchor_unsafe")

    def test_middle_threshold_with_overlapping_scores_is_not_healthy(self) -> None:
        rows = [
            {
                "threshold": 0.50,
                "pose_weighted_recall": 0.998,
                "weighted_recall_lower_confidence_bound": 0.994,
                "scoreDistribution": {
                    "positiveNegativeGapQ05Q95": -0.02,
                },
            },
        ]
        health = summarize_threshold_health(rows, 0.99, 0.99)
        self.assertFalse(health["hasHealthySafeWorkpoint"])
        self.assertTrue(health["safeButOverlappingScoreDistribution"])

    def test_healthy_selector_rejects_low_threshold_rescue(self) -> None:
        rows = [
            {"threshold": 0.10, "pose_weighted_recall": 0.999, "weighted_recall_lower_confidence_bound": 0.995, "scoreDistribution": {"positiveNegativeGapQ05Q95": 0.2}},
            {"threshold": 0.50, "pose_weighted_recall": 0.987, "weighted_recall_lower_confidence_bound": 0.91},
        ]
        self.assertIsNone(select_healthy_weighted_cull_workpoint(rows, 0.99, 0.99))


if __name__ == "__main__":
    unittest.main()
