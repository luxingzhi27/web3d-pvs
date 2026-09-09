from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

BENCHMARK = Path(__file__).resolve().parents[1]
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))

from plot_threshold_curves import load_validation_curve, parse_curve  # noqa: E402


class PlotThresholdCurvesTests(unittest.TestCase):
    def test_loads_only_test_free_validation_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "curve.json"
            path.write_text(
                json.dumps({"testRead": False, "rows": [
                    {"split": "calibration"},
                    {"split": "validation", "weighted_recall": 1.0},
                ]}),
                encoding="utf-8",
            )
            self.assertEqual(load_validation_curve(path), [{"split": "validation", "weighted_recall": 1.0}])
            path.write_text(json.dumps({"testRead": True, "rows": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not test-free"):
                load_validation_curve(path)

    def test_curve_argument_uses_label_and_path(self) -> None:
        label, path = parse_curve("Full=curve.json")
        self.assertEqual(label, "Full")
        self.assertEqual(path.name, "curve.json")


if __name__ == "__main__":
    unittest.main()
