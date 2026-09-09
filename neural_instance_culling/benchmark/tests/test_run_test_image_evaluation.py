from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

TESTS_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = TESTS_DIR.parent
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(BENCHMARK_DIR))

from test_instance_id_render_schema import make_formal_fixture  # noqa: E402
from run_test_image_evaluation import (  # noqa: E402
    renderer_command,
    run_formal_renderer,
    validate_test_manifest,
)


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


def write_fake_renderer(path: Path, *, formal_ready: bool = True, hardware: bool = True) -> None:
    renderer = f"""
import fs from 'node:fs';
import path from 'node:path';
const args = process.argv;
const manifestPath = args[args.indexOf('--manifest') + 1];
const outputDir = args[args.indexOf('--output-dir') + 1];
const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
fs.mkdirSync(outputDir, {{ recursive: true }});
fs.writeFileSync(path.join(outputDir, 'render_summary.json'), JSON.stringify({{
  schema: 'local-true-component-id-browser-summary-v3',
  renderStatus: 'rendered_component_id_buffers',
  componentIdShaderImplemented: true,
  formalImageEvaluationReady: {str(formal_ready).lower()},
  gpuBackend: {{ api: 'WebGL', vendor: 'Google Inc.', renderer: '{'ANGLE (NVIDIA, Vulkan)' if hardware else 'SwiftShader'}', version: 'WebGL 2.0' }},
  gpuGate: {{ required: true, hardware: {str(hardware).lower()} }},
  sampleCount: manifest.samples.length
}}, null, 2));
setTimeout(() => process.exit(0), 1200);
"""
    path.write_text(renderer, encoding="utf-8")


def write_fake_nvidia_smi(directory: Path, *, available: bool = True) -> None:
    directory.mkdir()
    script = directory / "nvidia-smi"
    script.write_text(
        "#!/bin/sh\n"
        + ("exit 1\n" if not available else
           "if [ \"$1\" = \"pmon\" ]; then printf '# gpu pid type sm mem enc dec command\\n 0 123 C 1 1 - - fake\\n'; else printf '0, NVIDIA Test GPU, 1.0, 0, 1\\n'; fi\n"),
        encoding="utf-8",
    )
    script.chmod(0o755)


def write_header_only_nvidia_smi(directory: Path) -> None:
    directory.mkdir()
    script = directory / "nvidia-smi"
    script.write_text(
        "#!/bin/sh\n"
        + "printf '# gpu pid type sm mem enc dec command\\n'\n",
        encoding="utf-8",
    )
    script.chmod(0o755)


class FrozenTestImageEvaluationTests(unittest.TestCase):
    def test_accepts_one_calibration_frozen_selection_kind(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = make_frozen_test_fixture(root)
            baseline.pop("threshold", None)
            baseline.pop("thresholdSelection")
            baseline.pop("thresholdProvenance")
            source_result = root / "region66.json"
            source_result.write_text(
                json.dumps({
                    "schema": "geometry-shell-hzb-browser-result-v1",
                    "mode": "Region66",
                    "formalReady": True,
                    "executionClass": "formal-hardware-gpu",
                    "gpuGate": {"required": True, "hardware": True},
                }),
                encoding="utf-8",
            )
            baseline["baselineSelection"] = {
                "method": "geometry-shell-hzb",
                "selectionSplit": "calibration",
                "testRead": False,
                "assetVariant": "equal-asset",
                "resolution": [128, 72],
                "depthBiasM": 0.001,
                "regionSampleCount": 0,
                "sourceResult": str(source_result.resolve()),
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

    def test_baseline_source_result_must_be_absolute_parseable_region66_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = make_frozen_test_fixture(root)
            manifest.pop("threshold", None)
            manifest.pop("thresholdSelection")
            manifest.pop("thresholdProvenance")
            manifest["baselineSelection"] = {
                "method": "geometry-shell-hzb",
                "selectionSplit": "calibration",
                "testRead": False,
                "assetVariant": "equal-asset",
                "resolution": [128, 72],
                "depthBiasM": 0.001,
                "regionSampleCount": 0,
                "sourceResult": "region66.json",
            }
            with self.assertRaisesRegex(ValueError, "absolute path"):
                validate_test_manifest(manifest)

            source = root / "region66.json"
            source.write_text(json.dumps({"schema": "wrong", "mode": "Region66"}), encoding="utf-8")
            manifest["baselineSelection"]["sourceResult"] = str(source.resolve())
            with self.assertRaisesRegex(ValueError, "formal Region66 HZB result"):
                validate_test_manifest(manifest)

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

    def test_formal_renderer_records_one_complete_host_evidence_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = make_frozen_test_fixture(root)
            renderer = root / "fake-renderer.mjs"
            write_fake_renderer(renderer)
            fake_bin = root / "bin"
            write_fake_nvidia_smi(fake_bin)
            output = root / "render-output"
            command = renderer_command(
                root / "manifest.json",
                output,
                renderer,
                None,
                False,
                None,
            )
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            old_path = os.environ.get("PATH", "")
            with mock.patch.dict(os.environ, {"PATH": f"{fake_bin}{os.pathsep}{old_path}"}):
                summary = run_formal_renderer(command, output, manifest, None)
            evidence = summary["hardwareEvidence"]
            self.assertTrue(evidence["complete"])
            self.assertTrue(evidence["formalReady"])
            self.assertTrue((output / "gpu_evidence.json").is_file())
            self.assertEqual(set(evidence["phases"]), {"before", "during", "after"})
            for phase in ("before", "during", "after"):
                self.assertTrue((output / "gpu_evidence" / f"nvidia_smi_{phase}.txt").is_file())
                self.assertTrue((output / "gpu_evidence" / f"nvidia_smi_pmon_{phase}.txt").is_file())
            self.assertTrue(summary["gpuGate"]["hostEvidenceComplete"])

    def test_formal_renderer_rejects_incomplete_or_software_execution(self) -> None:
        cases = (
            ("missing host evidence", {"available": False}, True, True, "before/during/after"),
            ("software backend", {"available": True}, True, False, "hardware WebGL"),
            ("formal-ready false", {"available": True}, False, True, "formalImageEvaluationReady=true"),
        )
        for name, nvidia_options, formal_ready, hardware, message in cases:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    manifest = make_frozen_test_fixture(root)
                    manifest_path = root / "manifest.json"
                    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                    renderer = root / "fake-renderer.mjs"
                    write_fake_renderer(renderer, formal_ready=formal_ready, hardware=hardware)
                    fake_bin = root / "bin"
                    write_fake_nvidia_smi(fake_bin, **nvidia_options)
                    output = root / "render-output"
                    command = renderer_command(manifest_path, output, renderer, None, False, None)
                    old_path = os.environ.get("PATH", "")
                    with mock.patch.dict(os.environ, {"PATH": f"{fake_bin}{os.pathsep}{old_path}"}):
                        with self.assertRaisesRegex(RuntimeError, message):
                            run_formal_renderer(command, output, manifest, None)

    def test_formal_renderer_rejects_header_only_pmon_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = make_frozen_test_fixture(root)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            renderer = root / "fake-renderer.mjs"
            write_fake_renderer(renderer)
            fake_bin = root / "bin"
            write_header_only_nvidia_smi(fake_bin)
            output = root / "render-output"
            command = renderer_command(manifest_path, output, renderer, None, False, None)
            old_path = os.environ.get("PATH", "")
            with mock.patch.dict(os.environ, {"PATH": f"{fake_bin}{os.pathsep}{old_path}"}):
                with self.assertRaisesRegex(RuntimeError, "before/during/after"):
                    run_formal_renderer(command, output, manifest, None)


if __name__ == "__main__":
    unittest.main()
