from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = TESTS_DIR.parent
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(BENCHMARK_DIR))

from test_run_test_image_evaluation import make_frozen_test_fixture  # noqa: E402
from select_qualitative_image_samples import (  # noqa: E402
    REGISTRATION_SCHEMA,
    select_qualitative_samples,
)


def make_selection_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    manifest = make_frozen_test_fixture(root)
    samples = []
    for index, source in enumerate(manifest["samples"] + manifest["samples"]):
        sample = copy.deepcopy(source)
        sample["sampleId"] = f"sample-{index}"
        sample["poseIndex"] = 200 + index
        samples.append(sample)
    manifest["samples"] = samples
    manifest["testCoverage"]["sampleCount"] = len(samples)
    manifest["testCoverage"]["uniquePoseCount"] = len(samples)
    manifest["subposeSelection"]["selectedSubposeCount"] = len(samples)
    manifest_path = root / "formal-test-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    metrics_path = root / "sample_image_metrics.json"
    metrics_path.write_text(
        json.dumps([
            {"sampleId": sample["sampleId"], "imageMetrics": {"PER": per}}
            for sample, per in zip(samples, [0.0, 0.1, 0.3, 0.9])
        ]),
        encoding="utf-8",
    )
    summary_path = root / "render_summary.json"
    summary_path.write_text("{}", encoding="utf-8")
    registration_path = root / "qualitative-registration.json"
    registration_path.write_text(
        json.dumps(
            {
                "schema": REGISTRATION_SCHEMA,
                "registeredBeforeTest": True,
                "selectionSplit": "pre_test",
                "testRead": False,
                "testEvaluationCount": 0,
                "poses": [
                    {"role": "fine_component", "poseIndex": 202},
                    {"role": "occlusion_boundary", "sampleId": "sample-3"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return summary_path, metrics_path, manifest_path, registration_path


class QualitativeImageSelectionTests(unittest.TestCase):
    def test_selects_percentiles_and_reuses_manifest_prediction_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary, metrics, manifest, registration = make_selection_fixture(root)
            result = select_qualitative_samples(
                summary, metrics, manifest, registration, root / "selection.json"
            )
            self.assertEqual(result["percentiles"]["median"], 0.2)
            self.assertAlmostEqual(result["percentiles"]["p95"], 0.81)
            self.assertEqual(
                [row["role"] for row in result["selections"]],
                ["median", "p95", "max_error", "fine_component", "occlusion_boundary"],
            )
            self.assertEqual(result["selections"][0]["predictionKey"], "vc00007")
            self.assertEqual(set(result["selections"][0]["images"]), {"Reference", "Prediction", "Difference"})
            self.assertNotIn("artifactsReady", result["selections"][0])

    def test_registration_must_precede_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary, metrics, manifest, registration = make_selection_fixture(root)
            value = json.loads(registration.read_text(encoding="utf-8"))
            value["testRead"] = True
            registration.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "precede test"):
                select_qualitative_samples(
                    summary, metrics, manifest, registration, root / "selection.json"
                )


if __name__ == "__main__":
    unittest.main()
