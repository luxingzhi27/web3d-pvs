from __future__ import annotations

import array
import copy
import contextlib
import io
import json
import struct
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
    scene = root.name
    runtime_meta_path = root / "runtimeVisibilityMeta.json"
    runtime_meta_path.write_text(
        json.dumps({"sceneName": scene, "instanceCount": 3}), encoding="utf-8"
    )
    base.update({
        "scene": scene,
        "runtimeMeta": str(runtime_meta_path.resolve()),
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
    poses = bytearray(8 * 64)
    poses[7 * 64 + 44] = 3
    struct.pack_into("<ff", poses, 7 * 64 + 36, 16.0 / 9.0, 1.0)
    (dataset / "poses.bin").write_bytes(poses)
    (dataset / "dataset_meta.json").write_text(json.dumps({
        "schema": "pose-csr-explicit-four-way-split-v1",
        "runtimeMeta": str(runtime_meta_path.resolve()),
        "poseCount": 8,
        "numInstances": 3,
        "candidateCount": 3,
        "modelInputFovYDeg": 66,
        "frontendRenderFovYDeg": 60,
        "splitIds": {"train": 0, "test": 3},
        "splitCounts": {"train": 7, "test": 1},
        "candidateSemantics": "test candidate semantics",
    }), encoding="utf-8")
    region_sampling = {"schema": "geometry-shell-hzb-region-sampling-v1", "requestedCount": 5}
    pose_selection = {
        "schema": "geometry-shell-hzb-pose-selection-v1",
        "split": "test",
        "selectedPoseCount": 1,
        "selectedPoseIndices": [7],
        "limit": 0,
        "representative": True,
    }
    pose_aspects = [{"poseId": 7, "aspect": 16.0 / 9.0, "fovYDeg": 66, "candidateCount": 3}]
    result = {
        "schema": "geometry-shell-hzb-browser-result-v1", "mode": "Region66",
        "formalReady": True, "executionClass": "formal-hardware-gpu",
        "gpuGate": {"required": True, "hardware": True},
        "datasetDir": str(dataset.resolve()),
        "workload": {
            "schema": "geometry-shell-hzb-browser-workload-v1", "split": "test",
            "scene": scene, "poseCount": 1, "candidateCount": 3,
            "width": 128, "height": 72, "fovYDeg": 66,
            "poseSelection": pose_selection, "warmupCount": 0,
            "source": {"datasetDirName": dataset.name, "candidateSemantics": "test candidate semantics"},
            "provenance": {
                "configuration": {"fovYDeg": 60, "regionFovYDeg": 66},
                "poseSelection": pose_selection,
                "poseAspects": pose_aspects,
            },
            "poses": pose_aspects, "regionSampling": region_sampling,
        },
        "regionSampling": region_sampling,
        "samples": [{
            "poseId": 7, "candidateCount": 3, "visibleInstanceIds": [2, 0],
            "timings": {"depthBiasM": 0.001},
        }],
    }
    (root / "region66-test.json").write_text(json.dumps(result), encoding="utf-8")
    return base, result, dataset


def convert(base: dict, result: dict, dataset: Path) -> dict:
    source = dataset.parent / "region66-test.json"
    source.write_text(json.dumps(result), encoding="utf-8")
    return build_hzb_image_manifest(
        base, result, candidate_dataset_dir=dataset,
        asset_variant="equal-asset", source_result=source,
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
                "regionSampleCount": 5,
                "sourceResult": str((Path(directory) / "region66-test.json").resolve()),
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
            candidate_meta = json.loads((dataset / "dataset_meta.json").read_text(encoding="utf-8"))
            candidate_meta["candidateCount"] = 2
            (dataset / "dataset_meta.json").write_text(json.dumps(candidate_meta), encoding="utf-8")
            outside_candidate = copy.deepcopy(result)
            outside_candidate["samples"][0].update(candidateCount=2, visibleInstanceIds=[1])
            outside_candidate["workload"]["poses"][0]["candidateCount"] = 2
            with self.assertRaisesRegex(ValueError, "outside its candidate set"):
                convert(base, outside_candidate, dataset)

            write_array(dataset / "candidate_offsets.bin", "Q", [0, 0, 0, 0, 0, 0, 0, 0, 3])
            write_array(dataset / "candidate_ids.bin", "I", [0, 1, 2])
            candidate_meta["candidateCount"] = 3
            (dataset / "dataset_meta.json").write_text(json.dumps(candidate_meta), encoding="utf-8")
            outside_component = copy.deepcopy(result)
            outside_component["samples"][0]["visibleInstanceIds"] = [3]
            with self.assertRaisesRegex(ValueError, "outside the instance range"):
                convert(base, outside_component, dataset)

    def test_rejects_scene_fov_aspect_and_candidate_provenance_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, result, dataset = make_inputs(root)

            wrong_scene = copy.deepcopy(result)
            wrong_scene["workload"]["scene"] = "other-scene"
            with self.assertRaisesRegex(ValueError, "does not match base scene"):
                convert(base, wrong_scene, dataset)

            wrong_fov = copy.deepcopy(result)
            wrong_fov["workload"]["fovYDeg"] = 60
            with self.assertRaisesRegex(ValueError, "workload fovYDeg"):
                convert(base, wrong_fov, dataset)

            wrong_aspect = copy.deepcopy(result)
            wrong_aspect["workload"]["poses"][0]["aspect"] = 1.0
            with self.assertRaisesRegex(ValueError, "aspect does not match"):
                convert(base, wrong_aspect, dataset)

            wrong_dataset = copy.deepcopy(result)
            wrong_dataset["workload"]["source"]["datasetDirName"] = "different-csr"
            with self.assertRaisesRegex(ValueError, "candidate dataset does not match"):
                convert(base, wrong_dataset, dataset)

            wrong_meta = json.loads((dataset / "dataset_meta.json").read_text(encoding="utf-8"))
            other_runtime_meta = root / "other-runtime-meta.json"
            other_runtime_meta.write_text(
                json.dumps({"sceneName": "other-scene", "instanceCount": 3}),
                encoding="utf-8",
            )
            wrong_meta["runtimeMeta"] = str(other_runtime_meta.resolve())
            (dataset / "dataset_meta.json").write_text(json.dumps(wrong_meta), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "runtime metadata scene does not match"):
                convert(base, result, dataset)

    def test_source_result_is_absolute_parseable_and_matches_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, result, dataset = make_inputs(root)
            output = convert(base, result, dataset)
            source_result = Path(output["baselineSelection"]["sourceResult"])
            self.assertTrue(source_result.is_absolute())
            self.assertEqual(json.loads(source_result.read_text(encoding="utf-8")), result)

            tampered = copy.deepcopy(result)
            tampered["samples"][0]["visibleInstanceIds"] = [1]
            with self.assertRaisesRegex(ValueError, "does not match the supplied HZB result"):
                build_hzb_image_manifest(
                    base,
                    tampered,
                    candidate_dataset_dir=dataset,
                    asset_variant="equal-asset",
                    source_result=source_result,
                )

    def test_accepts_result_level_dataset_provenance_when_workload_is_compact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, result, dataset = make_inputs(root)
            compact_result = copy.deepcopy(result)
            compact_result["workload"].pop("source")
            output = convert(base, compact_result, dataset)
            self.assertEqual(
                output["baselineSelection"]["sourceResult"],
                str((root / "region66-test.json").resolve()),
            )

    def test_accepts_compact_pose_aspects_provenance_without_repeated_fov(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, result, dataset = make_inputs(root)
            compact_result = copy.deepcopy(result)
            compact_result["workload"].pop("poses")
            compact_result["workload"]["provenance"]["poseAspects"][0].pop("fovYDeg")
            output = convert(base, compact_result, dataset)
            self.assertEqual(output[PREDICTION_COMPONENT_IDS_BY_KEY_FIELD], {"vc00007": [2, 0]})

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
