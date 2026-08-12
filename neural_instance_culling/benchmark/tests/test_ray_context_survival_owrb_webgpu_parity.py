from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmark"
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))

from prepare_ray_context_survival_owrb_webgpu_parity import _build_model  # noqa: E402
from ray_context_survival_owrb_model import RayContextSurvivalOWRBModel  # noqa: E402
from validate_ray_context_survival_owrb import _validate_webgpu_parity  # noqa: E402
from validate_ray_context_survival_owrb_webgpu_parity import validate  # noqa: E402


OUTPUT_NAMES = (
    "visibilityLogits",
    "visibilityScores",
    "utilityLogits",
    "utilityScores",
    "downloadLogits",
    "downloadScores",
)


class RayContextSurvivalOWRBBrowserParityTest(unittest.TestCase):
    def test_formal_summary_accepts_software_numeric_parity_but_keeps_hardware_claim_false(self) -> None:
        result = _validate_webgpu_parity(
            {
                "schema": "ray-context-survival-owrb-webgpu-parity-validation-v1",
                "status": "software_numeric_parity_passed",
                "passed": True,
                "backend": "webgpu",
                "formalReady": False,
                "hardwareClaim": "software WebGPU numeric parity only; no hardware WebGPU claim",
                "caseCount": 2,
                "maxAbs": 1e-6,
                "maxRelative": 2e-6,
            },
            require_artifact_files=False,
        )
        self.assertEqual(result["status"], "software_numeric_parity_passed")
        self.assertFalse(result["formalReady"])

    def test_formal_summary_rejects_missing_parity(self) -> None:
        with self.assertRaises(ValueError):
            _validate_webgpu_parity(None, require_artifact_files=False)

    def test_reference_reapplies_export_fp16_aabbs_after_checkpoint_load(self) -> None:
        """The host reference must use the same AABBs as the browser buffer."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runtime_meta = {
                "sceneBounds": {"min": [0.0, 0.0, 0.0], "max": [20.0, 20.0, 20.0]},
                "componentRecords": [
                    {
                        "componentGlobalId": 0,
                        "globalGlbId": 0,
                        "bounds": {
                            "min": [0.1234567, 1.234567, 2.345678],
                            "max": [3.456789, 4.567891, 5.678912],
                        },
                    },
                    {
                        "componentGlobalId": 1,
                        "globalGlbId": 0,
                        "bounds": {
                            "min": [6.1234567, 7.234567, 8.345678],
                            "max": [9.456789, 10.567891, 11.678912],
                        },
                    },
                ],
            }
            runtime_meta_path = root / "runtimeVisibilityMeta.json"
            runtime_meta_path.write_text(json.dumps(runtime_meta), encoding="utf-8")

            template = RayContextSurvivalOWRBModel(
                num_instances=2,
                num_glbs=1,
                point_hidden_dim=8,
                pointnetpp_centers=2,
                pointnetpp_neighbors=2,
                relation_hidden_dim=8,
            )
            checkpoint_path = root / "checkpoint.pt"
            torch.save({"config": template.config, "model": template.state_dict()}, checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

            reference, browser_aabbs = _build_model(checkpoint, runtime_meta_path)
            expected = np.asarray(
                [
                    [0.1234567, 1.234567, 2.345678, 3.456789, 4.567891, 5.678912],
                    [6.1234567, 7.234567, 8.345678, 9.456789, 10.567891, 11.678912],
                ],
                dtype=np.float16,
            ).astype(np.float32)
            np.testing.assert_array_equal(browser_aabbs, expected)
            np.testing.assert_array_equal(
                reference.instance_world_aabbs.detach().cpu().numpy(), expected
            )

    def test_shader_zeroes_survival_semantics_when_factor_is_disabled(self) -> None:
        shader_source = (
            ROOT.parent
            / "slm2viewer"
            / "scripts"
            / "benchmark_ray_context_survival_owrb_webgpu_parity.mjs"
        ).read_text(encoding="utf-8")
        self.assertIn("if (survivalEnabled) {", shader_source)
        self.assertIn(
            "semantic[semanticIndex] = 0.0;",
            shader_source,
        )

    def test_validator_accepts_float_bit_exact_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            expected_values = {
                name: [0.1, -0.2]
                for name in OUTPUT_NAMES
            }
            cases = {"schema": "ray-context-survival-owrb-webgpu-parity-cases-v1", "cases": [{"caseId": 0, "candidateIds": [2, 5], "expected": expected_values}]}
            values = np.asarray(
                [
                    expected_values[name][candidate]
                    for candidate in range(2)
                    for name in OUTPUT_NAMES
                ],
                dtype="<f4",
            )
            raw_words = values.view("<u4").astype(int).tolist()
            capture = {
                "schema": "ray-context-survival-owrb-webgpu-parity-capture-v1",
                "backend": "webgpu",
                "formalReady": True,
                "gpuGate": {"hardware": True},
                "results": [{"caseId": 0, "candidateIds": [2, 5], "rawOutput": raw_words, "forwardMs": 1.0}],
            }
            cases_path = root / "cases.json"
            capture_path = root / "capture.json"
            cases_path.write_text(json.dumps(cases), encoding="utf-8")
            capture_path.write_text(json.dumps(capture), encoding="utf-8")
            result = validate(capture_path, cases_path, atol=1e-6, rtol=1e-6, require_formal_gpu=True)
            self.assertTrue(result["passed"])
            self.assertEqual(result["caseCount"], 1)

    def test_validator_rejects_output_outside_elementwise_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            expected_values = {name: [0.0] for name in OUTPUT_NAMES}
            cases = {
                "schema": "ray-context-survival-owrb-webgpu-parity-cases-v1",
                "cases": [{"caseId": 0, "candidateIds": [7], "expected": expected_values}],
            }
            values = np.asarray([0.25 for _ in OUTPUT_NAMES], dtype="<f4")
            capture = {
                "schema": "ray-context-survival-owrb-webgpu-parity-capture-v1",
                "backend": "webgpu",
                "formalReady": False,
                "gpuGate": {"hardware": False},
                "results": [{
                    "caseId": 0,
                    "candidateIds": [7],
                    "rawOutput": values.view("<u4").astype(int).tolist(),
                }],
            }
            cases_path = root / "cases.json"
            capture_path = root / "capture.json"
            cases_path.write_text(json.dumps(cases), encoding="utf-8")
            capture_path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate(capture_path, cases_path, atol=1e-3, rtol=1e-3, require_formal_gpu=False)


if __name__ == "__main__":
    unittest.main()
