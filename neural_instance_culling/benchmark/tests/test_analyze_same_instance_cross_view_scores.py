from __future__ import annotations

from pathlib import Path
import sys
import unittest


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from analyze_same_instance_cross_view_scores import analyze_capture  # noqa: E402


class SameInstanceCrossViewScoreTests(unittest.TestCase):
    def _capture(self) -> dict:
        return {
            "split": "validation",
            "testRead": False,
            "threshold": 0.5,
            "perPose": [
                {
                    "candidateIds": [1, 2, 3],
                    "candidateScores": [0.9, 0.8, 0.1],
                    "targets": [1, 0, 1],
                    "visibleWeights": [4.0, 0.0, 1.0],
                },
                {
                    "candidateIds": [1, 2, 3],
                    "candidateScores": [0.6, 0.7, 0.2],
                    "targets": [0, 1, 1],
                    "visibleWeights": [0.0, 9.0, 1.0],
                },
            ],
        }

    def test_same_instance_ordering_and_false_positive_source(self) -> None:
        result = analyze_capture(self._capture())
        self.assertEqual(result["mixedLabelInstanceCount"], 2)
        self.assertEqual(result["sameInstancePairCount"], 2)
        self.assertEqual(result["sameInstancePairAuc"], 0.5)
        self.assertAlmostEqual(
            result["visibleWeightSameInstancePairAuc"], 2.0 / 5.0
        )
        self.assertEqual(result["meanOrderedInstanceFraction"], 0.5)
        self.assertEqual(result["falsePositiveCount"], 2)
        self.assertEqual(
            result["falsePositiveFromOtherVisibleViewFraction"], 1.0
        )

    def test_missing_persisted_ids_or_scores_is_rejected(self) -> None:
        capture = self._capture()
        capture["perPose"][0]["candidateIds"] = None
        with self.assertRaisesRegex(ValueError, "persist-ids"):
            analyze_capture(capture)

    def test_capture_without_mixed_instances_is_rejected(self) -> None:
        capture = self._capture()
        capture["perPose"][1]["targets"] = [1, 0, 1]
        with self.assertRaisesRegex(ValueError, "opposite labels"):
            analyze_capture(capture)


if __name__ == "__main__":
    unittest.main()
