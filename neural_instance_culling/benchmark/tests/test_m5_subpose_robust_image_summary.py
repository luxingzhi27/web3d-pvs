from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from neural_instance_culling.benchmark.summarize_m5_subpose_robust_image import summarize, write_report


class M5SubposeRobustImageSummaryTest(unittest.TestCase):
    def test_requires_strict_calibration_and_preserves_hardware_gate(self) -> None:
        variant = "pvs_m5_subpose_robust_v1_hkust_spatial_fov66"
        seeds = (20260801, 20260802, 20260803)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            image = root / "image"
            models = root / "models"
            image.mkdir()
            (image / "nvidia_smi_snapshot_test.csv").write_text("gpu,util\n0,99\n", encoding="utf-8")
            (image / "nvidia_smi_pmon_snapshot_test.txt").write_text("# gpu pid type sm\n0 1 C+G 99\n", encoding="utf-8")
            batches = []
            for seed in seeds:
                strict = models / f"{variant}_seed{seed}_full40_strict_calibration"
                strict.mkdir(parents=True)
                (strict / "best.pt").write_bytes(b"checkpoint")
                (strict / "calibration_ready_summary.json").write_text(
                    json.dumps(
                        {
                            "protocol": "calibration_ready_pre_test",
                            "frozenThreshold": 0.12,
                            "testEvaluationCount": 0,
                            "calibration": {
                                "selectionStatus": "safe",
                                "selected": {
                                    "pose_recall": 0.96,
                                    "pose_weighted_recall": 0.995,
                                    "weighted_recall_lower_confidence_bound": 0.992,
                                },
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                for split in ("validation", "calibration"):
                    batches.append(
                        {
                            "batchId": f"{variant}_seed{seed}_{split}",
                            "sampleCount": 2,
                            "imageMetrics": {
                                "totalPixels": 100,
                                "validReferencePixels": 80,
                                "backgroundReferencePixels": 20,
                                "errorPixels": 1,
                                "missPixels": 1,
                                "wrongInstancePixels": 0,
                                "extraPixels": 0,
                                "evaluatedSubposeCount": 2,
                                "renderFailedSubposeCount": 0,
                                "missingGlbSubposeCount": 0,
                            },
                            "viewCellImageMetrics": {
                                "viewCellMissPixelRateMean": 0.0125,
                                "viewCellMissPixelRateP95": 0.0125,
                            },
                        }
                    )
            (image / "summary.json").write_text(
                json.dumps(
                    {
                        "testRead": False,
                        "formalImageEvaluationReady": False,
                        "renderFovYDeg": 60,
                        "modelInputFovYDeg": 66,
                        "browserGpuBackend": {
                            "vendor": "Google Inc. (NVIDIA)",
                            "renderer": "ANGLE (NVIDIA, Vulkan RTX A6000)",
                            "version": "WebGL 2.0",
                        },
                        "browserGpuGate": {
                            "required": True,
                            "hardware": True,
                            "softwareMarkers": None,
                        },
                        "batches": batches,
                    }
                ),
                encoding="utf-8",
            )
            summary = summarize(image, models, variant, seeds)
            self.assertEqual(summary["imageMetrics"]["sampleCount"], 12)
            self.assertEqual(summary["qualityGate"]["status"], "No-Go")
            report = root / "report.md"
            write_report(summary, report)
            self.assertIn("硬件 GPU", report.read_text(encoding="utf-8"))

    def test_self_test_contract_is_strict_about_batch_ids(self) -> None:
        # The public self-test is intentionally small; this assertion protects
        # the registered naming contract used by the formal wrapper.
        from neural_instance_culling.benchmark.summarize_m5_subpose_robust_image import BATCH_RE

        match = BATCH_RE.match("pvs_m5_subpose_robust_v1_hkust_spatial_fov66_seed20260801_validation")
        self.assertIsNotNone(match)
        self.assertEqual(match.group("seed"), "20260801")
        self.assertIsNone(BATCH_RE.match("pvs_m5_subpose_robust_v1_hkust_spatial_fov66_validation"))


if __name__ == "__main__":
    unittest.main()
