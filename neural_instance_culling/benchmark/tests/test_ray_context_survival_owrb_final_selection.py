from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from select_ray_context_survival_owrb_final import select_final_member  # noqa: E402


class RayContextSurvivalOWRBFinalSelectionTest(unittest.TestCase):
    def test_selection_reads_only_validation_summary(self) -> None:
        summary = {
            "schema": "ray-context-survival-owrb-matrix-summary-v1",
            "testRead": False,
            "variants": ["a", "b"],
            "seeds": [1, 2, 3],
            "members": {"a": {}, "b": {}},
        }
        for variant in summary["variants"]:
            for seed in summary["seeds"]:
                summary["members"][variant][str(seed)] = {
                    "safeWorkpoint": True,
                    "threshold": 0.5,
                    "checkpoint": f"{variant}-{seed}.pt",
                    "runtimeFeatures": f"{variant}-{seed}.bin",
                    "thresholdSource": {"safeWorkpoint": True},
                    "poseMacro": {
                        "weightedRecall": 0.999,
                        "usefulCull": 0.3 if variant == "a" else 0.2,
                        "balancedAccuracy": 0.8,
                        "precision": 0.7,
                    },
                    "aggregate": {"avgPredCount": 10.0, "weightedRecall": 0.999},
                }
        result = select_final_member(summary, Path("summary.json"))
        self.assertEqual(result["status"], "selected_validation_safe_final_member")
        self.assertEqual(result["selection"]["variant"], "a")
        self.assertFalse(result["testRead"])

    def test_selection_rejects_test_read_summary(self) -> None:
        with self.assertRaises(ValueError):
            select_final_member({"schema": "ray-context-survival-owrb-matrix-summary-v1", "testRead": True}, Path("summary.json"))

    def test_selection_requires_aggregate_weighted_recall(self) -> None:
        summary = {
            "schema": "ray-context-survival-owrb-matrix-summary-v1",
            "testRead": False,
            "variants": ["a"],
            "seeds": [1],
            "members": {
                "a": {
                    "1": {
                        "safeWorkpoint": True,
                        "poseMacro": {"weightedRecall": 0.999, "usefulCull": 0.5},
                        "aggregate": {"weightedRecall": 0.989, "avgPredCount": 1.0},
                    }
                }
            },
        }
        result = select_final_member(summary, Path("summary.json"))
        self.assertEqual(result["status"], "no_validation_safe_final_member")


if __name__ == "__main__":
    unittest.main()
