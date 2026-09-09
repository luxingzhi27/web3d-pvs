from __future__ import annotations

import array
import copy
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = TESTS_DIR.parent
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(BENCHMARK_DIR))

from build_hzb_image_manifest import build_hzb_image_manifest, parse_args  # noqa: E402
from instance_id_render_schema import PREDICTION_COMPONENT_IDS_BY_KEY_FIELD  # noqa: E402
from test_instance_id_render_schema import make_formal_fixture  # noqa: E402


def write_array(path: Path, typecode: str, values: list[int]) -> None:
    with path.open("wb") as stream:
        array.array(typecode, values).tofile(stream)


def make_inputs(root: Path) -> tuple[dict, dict, Path]:
    base = make_formal_fixture(root)
    base.update({
        "split": "test", "testRead": True, "testEvaluationCount": 1,
        "threshold": 0.68,
        "thresholdSelection": {"threshold": 0.68, "selectionSplit": "calibration", "testRead": False},
        "thresholdProvenance": {"selectionSplit": "calibration", "testRead": False},
        "testCoverage": {
            "split": "test", "selection": "all_unique_test_viewcells", "viewcellCount": 1,
            "sampleCount": 2, "uniquePoseCount": 2, "maxViewcells": 0,
            "sampledWithReplacement": False, "subposesPerViewcell": 0,
        },
        "subposeSelection": {
            "mode": "all", "requestedPerViewcell": 0, "selectedSubposeCount": 2,
            "viewcellCount": 1,
        },
    })
    dataset = root / "candidate-csr"
    dataset.mkdir()
    write_array(dataset / "candidate_offsets.bin", "Q", [0, 0, 0, 0, 0, 0, 0, 0, 3])
    write_array(dataset / "candidate_ids.bin", "I", [0, 1, 2])
    region_sampling = {"schema": "geometry-shell-hzb-region-sampling-v1", "requestedCount": 5}
    result = {
        "schema": "geometry-shell-hzb-browser-result-v1", "mode": "Region66",
        "workload": {
            "schema": "geometry-shell-hzb-browser-workload-v1", "split": "test",
            "poseCount": 1, "width": 128, "height": 72, "regionSampling": region_sampling,
        },
        "regionSampling": region_sampling,
        "samples": [{
            "poseId": 7, "candidateCount": 3, "visibleInstanceIds": [2, 0],
            "timings": {"depthBiasM": 0.001},
        }],
    }
    return base, result, dataset


def convert(base: dict, result: dict, dataset: Path) -> dict:
    return build_hzb_image_manifest(
        base, result, candidate_dataset_dir=dataset,
        asset_variant="equal-asset", source_result="region66-test.json",
    )


class HzbImageManifestTests(unittest.TestCase):
    def test_conversion_replaces_keyed_ids_and_writes_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base, result, dataset = make_inputs(Path(directory))
            output = convert(base, result, dataset)
            self.assertEqual(output[PREDICTION_COMPONENT_IDS_BY_KEY_FIELD], {"vc00007": [2, 0]})
            self.assertEqual(output["samples"], base["samples"])
            self.assertEqual(output["baselineSelection"], {
                "method": "geometry-shell-hzb", "selectionSplit": "calibration", "testRead": False,
                "assetVariant": "equal-asset", "resolution": [128, 72], "depthBiasM": 0.001,
                "regionSampleCount": 5, "sourceResult": "region66-test.json",
            })
            for field in ("threshold", "thresholdSelection", "thresholdProvenance"):
                self.assertNotIn(field, output)

    def test_cli_requires_explicit_candidate_dataset(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--base-manifest", "base.json", "--hzb-result", "result.json",
                            "--asset-variant", "equal-asset", "--output", "out.json"])

    def test_rejects_nonformal_base_and_missing_or_duplicate_poses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base, result, dataset = make_inputs(Path(directory))
            invalid_base = copy.deepcopy(base)
            invalid_base["schema"] = "local-true-component-id-render-manifest-v3"
            with self.assertRaisesRegex(ValueError, "formal"):
                convert(invalid_base, result, dataset)

            missing = copy.deepcopy(result)
            missing["samples"][0]["poseId"] = 8
            with self.assertRaisesRegex(ValueError, "exactly cover"):
                convert(base, missing, dataset)

            duplicate = copy.deepcopy(result)
            duplicate["samples"].append(copy.deepcopy(duplicate["samples"][0]))
            duplicate["workload"]["poseCount"] = 2
            with self.assertRaisesRegex(ValueError, "duplicate poseId"):
                convert(base, duplicate, dataset)

    def test_rejects_calibration_result_and_predictions_outside_candidate_or_component(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base, result, dataset = make_inputs(Path(directory))
            calibration = copy.deepcopy(result)
            calibration["workload"]["split"] = "calibration"
            with self.assertRaisesRegex(ValueError, "test"):
                convert(base, calibration, dataset)

            write_array(dataset / "candidate_offsets.bin", "Q", [0, 0, 0, 0, 0, 0, 0, 0, 2])
            write_array(dataset / "candidate_ids.bin", "I", [0, 2])
            outside_candidate = copy.deepcopy(result)
            outside_candidate["samples"][0].update(candidateCount=2, visibleInstanceIds=[1])
            with self.assertRaisesRegex(ValueError, "outside its candidate set"):
                convert(base, outside_candidate, dataset)

            write_array(dataset / "candidate_offsets.bin", "Q", [0, 0, 0, 0, 0, 0, 0, 0, 3])
            write_array(dataset / "candidate_ids.bin", "I", [0, 1, 2])
            outside_component = copy.deepcopy(result)
            outside_component["samples"][0]["visibleInstanceIds"] = [3]
            with self.assertRaisesRegex(ValueError, "outside the instance range"):
                convert(base, outside_component, dataset)

    def test_converted_manifest_passes_schema_only_renderer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, result, dataset = make_inputs(root)
            manifest_path = root / "hzb-formal-test.json"
            manifest_path.write_text(json.dumps(convert(base, result, dataset)), encoding="utf-8")
            output_dir = root / "schema-output"
            completed = subprocess.run([
                sys.executable, str(BENCHMARK_DIR / "run_test_image_evaluation.py"),
                "--manifest", str(manifest_path), "--output-dir", str(output_dir),
                "--require-hardware-gpu", "--render-schema-only",
            ], capture_output=True, text=True, check=False, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads((output_dir / "render_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["validation"]["formalImageEvaluationReady"])
            self.assertEqual(summary["validation"]["predictionKeyCount"], 1)
            self.assertFalse((output_dir / "gpu_evidence").exists())


if __name__ == "__main__":
    unittest.main()
