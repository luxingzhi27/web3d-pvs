from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_DIR))

from audit_training_stability import audit_history  # noqa: E402


class TrainingStabilityAuditTests(unittest.TestCase):
    def _write(self, payload: object) -> Path:
        handle = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        path = Path(handle.name)
        with handle:
            handle.write(json.dumps(payload).encode("utf-8"))
        return path

    def test_sums_skipped_gradients_across_all_epochs(self) -> None:
        path = self._write([
            {"epoch": 1, "stepsPerEpoch": 10, "trainSkippedNonFiniteGrad": 2, "loss": 1.0},
            {"epoch": 2, "stepsPerEpoch": 10, "trainSkippedNonFiniteGrad": 0, "loss": 0.5},
        ])
        report = audit_history(path, expected_epochs=2)
        self.assertEqual(report["trainSkippedNonFiniteGradTotal"], 2)
        self.assertEqual(report["status"], "warning")
        self.assertAlmostEqual(report["gradientSkipRate"], 0.1)

    def test_nonfinite_loss_is_failure_even_if_last_epoch_is_finite(self) -> None:
        path = self._write([
            {"epoch": 1, "stepsPerEpoch": 10, "trainSkippedNonFiniteLoss": 1, "loss": "NaN"},
            {"epoch": 2, "stepsPerEpoch": 10, "trainSkippedNonFiniteLoss": 0, "loss": 0.5},
        ])
        report = audit_history(path, expected_epochs=2)
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["trainSkippedNonFiniteLossTotal"], 1)

    def test_epoch_sequence_is_strict(self) -> None:
        path = self._write([
            {"epoch": 1, "stepsPerEpoch": 1},
            {"epoch": 1, "stepsPerEpoch": 1},
        ])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            audit_history(path)


if __name__ == "__main__":
    unittest.main()
