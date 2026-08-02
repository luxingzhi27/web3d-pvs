from __future__ import annotations

import sys
import unittest
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_DIR))

import summarize_formal_m4_matrix as m4_summary  # noqa: E402


class FormalM4SummaryTests(unittest.TestCase):
    def test_pre_test_protocol_is_not_rejected_by_word_test(self) -> None:
        payload = {
            "split": "validation",
            "thresholdSource": {
                "protocol": "calibration_ready_pre_test",
                "testEvaluationCount": 0,
                "testThresholdOverride": False,
            },
        }
        m4_summary.validate_threshold_provenance(payload, Path("fixture.json"))

    def test_test_derived_provenance_is_rejected(self) -> None:
        for source in (
            {
                "protocol": "calibration_ready_pre_test",
                "testEvaluationCount": 1,
                "testThresholdOverride": False,
            },
            {
                "protocol": "frozen_calibration_one_shot_test",
                "testEvaluationCount": 1,
                "testThresholdOverride": False,
            },
            {
                "protocol": "calibration_ready_pre_test",
                "testEvaluationCount": 0,
                "testThresholdOverride": True,
            },
        ):
            with self.assertRaises(ValueError):
                m4_summary.validate_threshold_provenance(
                    {"split": "validation", "thresholdSource": source}, Path("fixture.json")
                )


if __name__ == "__main__":
    unittest.main()
