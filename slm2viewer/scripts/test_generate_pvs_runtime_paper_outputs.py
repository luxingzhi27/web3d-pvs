#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

from generate_pvs_runtime_paper_outputs import build_outputs


def upload(label: str, scene: str, backend: str, session_count: int) -> dict:
    candidates = [500, 1_500, 3_000, 7_000, 12_000, 16_000]
    sessions = []
    for session_index in range(session_count):
        samples = []
        for ordinal, candidate in enumerate(candidates):
            samples.append({
                "poseId": ordinal,
                "ordinal": ordinal,
                "candidateCount": candidate,
                "modelInferenceMs": 1.0 + candidate / 1_000.0 + session_index / 10.0,
                "gpuKernelMs": 1.0 + candidate / 1_000.0 + session_index / 10.0,
                "submitCompletionMs": 2.0 + candidate / 1_000.0,
                "timingSource": "webgpu-timestamp-query",
            })
        sessions.append({
            "sessionIndex": session_index,
            "orderSeed": 7 + session_index,
            "warmupCount": 50,
            "samples": samples,
        })
    return {
        "schema": "pvs-v4-browser-runtime-result-v1",
        "device": {"label": label, "model": "Test GPU"},
        "environment": {
            "secureContext": True,
            "backend": backend,
            "adapter": {"vendor": "test", "architecture": "gpu"},
            "hardwareGate": {"hardware": True},
            "wasmSimd": backend == "wasm-simd-v4",
            "visibilityViolations": 0,
        },
        "workload": {
            "scene": scene,
            "split": "test",
            "poseCount": len(candidates),
            "candidateCount": sum(candidates),
        },
        "model": {"runtimeAssetBytes": 1_048_576},
        "timingDefinition": {"primary": "gpuKernelMs"},
        "serverReceipt": {"formalReady": True},
        "sessions": sessions,
    }


class RuntimePaperOutputTest(unittest.TestCase):
    def test_formal_filter_stats_bins_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory) / "uploads"
            output_dir = Path(directory) / "outputs"
            input_dir.mkdir()
            (input_dir / "formal.json").write_text(
                json.dumps(upload("macbook air m2", "hkust-v3", "webgpu-v4", 5)),
                encoding="utf-8",
            )
            (input_dir / "smoke.json").write_text(
                json.dumps(upload("RTX A6000 smoke", "ifcbench", "wasm-simd-v4", 5)),
                encoding="utf-8",
            )

            summary = build_outputs([input_dir], output_dir, bootstrap_repetitions=100, seed=3)
            formal = [row for row in summary["groups"] if row["status"] == "formal"]
            self.assertEqual(len(formal), 1)
            row = formal[0]
            self.assertEqual(row["device"], "MacBook Air M2")
            self.assertEqual(row["scene"], "hkust")
            self.assertEqual(row["formalSessions"], 5)
            self.assertEqual(row["sampleCount"], 30)
            self.assertAlmostEqual(row["latencyMeanMs"], 7.8666666667)
            self.assertAlmostEqual(row["fitInterceptMs"], 1.2)
            self.assertAlmostEqual(row["fitSlopeMsPerCandidate"], 0.001)
            self.assertEqual(row["latencyP50Ci95LowMs"] <= row["latencyP50Ms"], True)
            self.assertEqual(row["latencyP50Ms"] <= row["latencyP50Ci95HighMs"], True)

            smoke = [item for item in summary["excludedInputs"] if item["file"] == "smoke.json"]
            self.assertEqual(len(smoke), 1)
            self.assertTrue(smoke[0]["reason"].startswith("smoke-excluded:"))
            unavailable = [
                item for item in summary["groups"]
                if item["device"] == "RTX A6000"
                and item["scene"] == "ifcbench"
                and item["backend"] == "wasm-simd-v4"
            ]
            self.assertEqual(unavailable[0]["status"], "unavailable")
            self.assertTrue(unavailable[0]["reason"].startswith("smoke-excluded:"))

            bins = [item for item in summary["candidateBuckets"] if item["status"] == "formal"]
            self.assertEqual([item["candidateBin"] for item in bins], [
                "0-1k", "1-2k", "2-5k", "5-10k", "10-15k", "15k+",
            ])
            self.assertTrue(all(item["sampleCount"] == 5 for item in bins))
            self.assertEqual(sum(1 for item in summary["groups"] if item["scene"] == "ifcbench"), 4)
            self.assertEqual(len(summary["formalInputs"]), 1)

            with (output_dir / "runtime_paper_source.csv").open(encoding="utf-8") as handle:
                self.assertEqual(sum(1 for _ in handle), 31)
            for suffix in ("csv", "pdf", "svg", "png"):
                if suffix == "csv":
                    self.assertTrue((output_dir / "runtime_paper_summary.csv").exists())
                else:
                    self.assertTrue((output_dir / f"runtime_paper_latency.{suffix}").exists())


if __name__ == "__main__":
    unittest.main()
