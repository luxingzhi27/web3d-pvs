from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

BENCHMARK = Path(__file__).resolve().parents[1]
MODEL = BENCHMARK.parent / "model"
for directory in (BENCHMARK, MODEL):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from model_runners import _v4_frozen_threshold  # noqa: E402


class ExactCalibrationThresholdTests(unittest.TestCase):
    def test_checkpoint_specific_exact_calibration_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "last.pt"
            checkpoint.write_bytes(b"checkpoint")
            calibration = root / "exact_calibration.json"
            calibration.write_text(json.dumps({
                "schema": "pvs-v4-exact-calibration-v2",
                "split": "calibration",
                "testRead": False,
                "checkpoint": str(checkpoint),
                "predictionRule": "score >= threshold",
                "status": "confidence_target_met",
                "selection": {"threshold": 0.625, "confidenceTargetMet": True},
                "selected": {
                    "threshold": 0.625,
                    "aggregateWeightedRecall": 0.995,
                    "aggregateWeightedRecallLowerConfidenceBound": 0.991,
                },
            }), encoding="utf-8")
            self.assertEqual(
                _v4_frozen_threshold({}, checkpoint, calibration),
                0.625,
            )

    def test_exact_calibration_must_match_checkpoint_and_mean_wr_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "last.pt"
            checkpoint.write_bytes(b"checkpoint")
            calibration = root / "exact_calibration.json"
            payload = {
                "schema": "pvs-v4-exact-calibration-v2",
                "split": "calibration",
                "testRead": False,
                "checkpoint": str(root / "other.pt"),
                "predictionRule": "score >= threshold",
                "status": "confidence_target_met",
                "selection": {"threshold": 0.625, "confidenceTargetMet": True},
                "selected": {
                    "threshold": 0.625,
                    "aggregateWeightedRecall": 0.995,
                    "aggregateWeightedRecallLowerConfidenceBound": 0.991,
                },
            }
            calibration.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different checkpoint"):
                _v4_frozen_threshold({}, checkpoint, calibration)
            payload["checkpoint"] = str(checkpoint)
            payload["status"] = "mean_target_met"
            payload["selection"]["confidenceTargetMet"] = False
            payload["selected"]["aggregateWeightedRecallLowerConfidenceBound"] = 0.989
            calibration.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(_v4_frozen_threshold({}, checkpoint, calibration), 0.625)
            payload["selected"]["aggregateWeightedRecall"] = 0.99
            calibration.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "weighted-recall target"):
                _v4_frozen_threshold({}, checkpoint, calibration)


if __name__ == "__main__":
    unittest.main()
