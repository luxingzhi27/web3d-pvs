from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "build_paper_result_bundle.py"
SPEC = importlib.util.spec_from_file_location("build_paper_result_bundle", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PaperResultBundleTest(unittest.TestCase):
    def test_ablation_mean_and_sample_std_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "ablation.json"
            metrics_a = {metric: 1.0 for metric in MODULE.ABLATION_METRICS}
            metrics_b = {metric: 3.0 for metric in MODULE.ABLATION_METRICS}
            source.write_text(json.dumps({
                "schema": "fixture",
                "split": "validation",
                "testRead": False,
                "reference": "full",
                "bootstrap": {"replicates": 10000},
                "pairedComparisons": {},
                "rows": {"full": {
                    "1": {
                        "validationWeightedRecallLowerConfidenceBound": 0.991,
                        "runtimeFeatureBytes": 1048576,
                        "metrics": metrics_a,
                    },
                    "2": {
                        "validationWeightedRecallLowerConfidenceBound": 0.989,
                        "runtimeFeatureBytes": 1048576,
                        "metrics": metrics_b,
                    },
                }},
            }), encoding="utf-8")

            MODULE.build_ablation(source, root / "paper")
            with (root / "paper/ablation/core_ablation.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["seedCount"], "2")
            self.assertEqual(row["safeSeedCount"], "1")
            self.assertEqual(float(row["posePrecisionMean"]), 2.0)
            self.assertAlmostEqual(float(row["posePrecisionStd"]), 2 ** 0.5)


if __name__ == "__main__":
    unittest.main()
