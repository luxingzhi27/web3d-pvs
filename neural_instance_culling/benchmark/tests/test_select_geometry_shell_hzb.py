from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


BENCHMARK = Path(__file__).resolve().parents[1]
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))

import select_geometry_shell_hzb as selector  # noqa: E402


class GeometryShellHZBSelectionTests(unittest.TestCase):
    def _candidate(
        self,
        root: Path,
        name: str,
        *,
        useful: float,
        recall: float = 0.991,
        lower: float = 0.9905,
        formal: bool = True,
    ) -> Path:
        shell = root / f"{name}_shell"
        shell.mkdir()
        (shell / "shell_meta.json").write_text(
            json.dumps({"variant": "lossless"}), encoding="utf-8"
        )
        result = root / f"{name}_result.json"
        result.write_text(
            json.dumps(
                {
                    "formalReady": formal,
                    "executionClass": "formal-hardware-gpu" if formal else "hardware-smoke-concurrent",
                    "shellDir": str(shell),
                    "workload": {
                        "width": 512,
                        "height": 288,
                        "depthBiasM": 0.001,
                        "regionSampling": {"requestedCount": 0},
                    },
                }
            ),
            encoding="utf-8",
        )
        metrics = root / f"{name}_metrics.json"
        metrics.write_text(
            json.dumps(
                {
                    "schema": selector.METRICS_SCHEMA,
                    "sourceResult": str(result),
                    "split": "calibration",
                    "mode": "Region66",
                    "weightedRecall": recall,
                    "weightedRecallLower95": lower,
                    "aggregate": {
                        "usefulCull": useful,
                        "badCull": 0.0001,
                        "balancedAccuracy": 0.8,
                        "specificity": 0.7,
                        "precision": 0.3,
                    },
                    "timing": {"totalP50Ms": 4.0},
                    "formalReady": formal,
                    "executionClass": "formal-hardware-gpu" if formal else "hardware-smoke-concurrent",
                    "gpuGate": {
                        "required": True,
                        "hardware": True,
                        "adapter": {"vendor": "nvidia"},
                    },
                    "gpuConcurrency": {"concurrentComputeDetected": False},
                }
            ),
            encoding="utf-8",
        )
        return metrics

    def test_safe_pool_is_ranked_by_useful_cull(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            low = self._candidate(root, "low", useful=0.4)
            high = self._candidate(root, "high", useful=0.7)
            payload = selector.select([low, high])
            self.assertEqual(payload["status"], "safe")
            self.assertEqual(payload["selected"]["usefulCull"], 0.7)
            self.assertFalse(payload["testRead"])

    def test_diagnostic_selection_does_not_claim_safety(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self._candidate(root, "first", useful=0.8, lower=0.98)
            second = self._candidate(root, "second", useful=0.2, lower=0.989)
            payload = selector.select([first, second])
            self.assertEqual(payload["status"], "no_qualified_safety_workpoint")
            self.assertEqual(payload["selected"]["weightedRecallLower95"], 0.989)
            self.assertFalse(payload["selected"]["safe"])

    def test_nonformal_results_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            candidate = self._candidate(
                Path(temporary), "smoke", useful=0.5, formal=False
            )
            with self.assertRaisesRegex(ValueError, "not a formal hardware run"):
                selector.select([candidate])


if __name__ == "__main__":
    unittest.main()
