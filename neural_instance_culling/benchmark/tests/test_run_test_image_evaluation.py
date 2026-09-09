from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = TESTS_DIR.parent
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(BENCHMARK_DIR))

from test_instance_id_render_schema import make_formal_fixture  # noqa: E402
from run_test_image_evaluation import validate_test_manifest  # noqa: E402


def make_frozen_test_fixture(root: Path) -> dict:
    manifest = make_formal_fixture(root)
    manifest.update(
        {
            "split": "test",
            "testRead": True,
            "testEvaluationCount": 1,
            "threshold": 0.68,
            "thresholdSelection": {
                "threshold": 0.68,
                "selectionSplit": "calibration",
                "testRead": False,
            },
            "thresholdProvenance": {
                "selectionSplit": "calibration",
                "testRead": False,
            },
            "testCoverage": {
                "split": "test",
                "selection": "all_unique_test_viewcells",
                "viewcellCount": 1,
                "sampleCount": 2,
                "uniquePoseCount": 2,
                "maxViewcells": 0,
                "sampledWithReplacement": False,
                "subposesPerViewcell": 0,
            },
            "subposeSelection": {
                "mode": "all",
                "requestedPerViewcell": 0,
                "selectedSubposeCount": 2,
                "viewcellCount": 1,
            },
        }
    )
    for index, sample in enumerate(manifest["samples"]):
        sample["poseIndex"] = 100 + index
    return manifest


class FrozenTestImageEvaluationTests(unittest.TestCase):
    def test_accepts_one_calibration_frozen_selection_kind(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            baseline = make_frozen_test_fixture(Path(directory))
            baseline.pop("threshold", None)
            baseline.pop("thresholdSelection")
            baseline.pop("thresholdProvenance")
            baseline["baselineSelection"] = {
                "method": "geometry-shell-hzb",
                "selectionSplit": "calibration",
                "testRead": False,
                "assetVariant": "equal-asset",
                "resolution": [128, 72],
                "depthBiasM": 0.001,
                "regionSampleCount": 0,
                "sourceResult": "region66.json",
            }
            validation = validate_test_manifest(baseline)
            self.assertEqual(validation["selectionMethod"], "geometry-shell-hzb")

            selected_from_test = copy.deepcopy(baseline)
            selected_from_test["baselineSelection"]["selectionSplit"] = "test"
            with self.assertRaisesRegex(ValueError, "calibration"):
                validate_test_manifest(selected_from_test)

            mixed = copy.deepcopy(baseline)
            mixed["thresholdSelection"] = {
                "threshold": 0.5,
                "selectionSplit": "calibration",
                "testRead": False,
            }
            with self.assertRaisesRegex(ValueError, "exactly one"):
                validate_test_manifest(mixed)

    def test_manifest_rejects_test_selection_or_incomplete_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = make_frozen_test_fixture(Path(directory))
            self.assertEqual(validate_test_manifest(manifest)["selectionSplit"], "calibration")

            selected_from_test = copy.deepcopy(manifest)
            selected_from_test["thresholdSelection"]["selectionSplit"] = "test"
            with self.assertRaisesRegex(ValueError, "calibration"):
                validate_test_manifest(selected_from_test)

            subset = copy.deepcopy(manifest)
            subset["testCoverage"]["subposesPerViewcell"] = 4
            with self.assertRaisesRegex(ValueError, "all dense subposes"):
                validate_test_manifest(subset)

            truncated = copy.deepcopy(manifest)
            truncated["testCoverage"]["selection"] = "selected_split_viewcells"
            with self.assertRaisesRegex(ValueError, "all unique"):
                validate_test_manifest(truncated)

    @unittest.skipUnless(shutil.which("node"), "Node.js is not installed")
    def test_schema_only_delegates_prediction_key_reuse_to_renderer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "formal-test.json"
            manifest_path.write_text(json.dumps(make_frozen_test_fixture(root)), encoding="utf-8")
            output = root / "render-output"
            result = subprocess.run(
                [
                    sys.executable,
                    str(BENCHMARK_DIR / "run_test_image_evaluation.py"),
                    "--manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(output),
                    "--require-hardware-gpu",
                    "--render-schema-only",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((output / "render_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["validation"]["predictionKeyReuseCount"], 1)
            self.assertFalse((output / "gpu_evidence").exists())


if __name__ == "__main__":
    unittest.main()
