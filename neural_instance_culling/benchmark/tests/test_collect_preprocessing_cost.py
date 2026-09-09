from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "collect_preprocessing_cost.py"
SPEC = importlib.util.spec_from_file_location("collect_preprocessing_cost", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PreprocessingCostCollectorTests(unittest.TestCase):
    def test_fixed_reader_uses_explicit_fields_and_real_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample.log").write_text(
                'WebGL backend vendor="fixture" renderer="GPU"\n{"elapsedMs": 1000}\n{"elapsedMs": 2000}\n',
                encoding="utf-8",
            )
            (root / "relation.jsonl").write_text(
                "\n".join(json.dumps(item) for item in [
                    {"returnCode": 1, "elapsedSeconds": 100},
                    {"returnCode": 0, "elapsedSeconds": 3},
                    {"returnCode": 0, "elapsedSeconds": 4},
                ]) + "\n",
                encoding="utf-8",
            )
            (root / "history_a.json").write_text(json.dumps([{"elapsedSeconds": 2}, {"elapsedSeconds": 5}]), encoding="utf-8")
            (root / "history_b.json").write_text(json.dumps([{"elapsedSeconds": 1.5}, {"elapsedSeconds": 4.5}]), encoding="utf-8")
            (root / "train.jsonl").write_text("\n".join(json.dumps(item) for item in [
                {"device": "NVIDIA fixture", "cudaPeakMemoryAllocatedMiB": 10},
                {"device": "NVIDIA fixture", "cudaPeakMemoryAllocatedMiB": 12.5},
            ]) + "\n", encoding="utf-8")
            (root / "output.bin").write_bytes(b"12345")
            original = MODULE.SCENE_SPECS
            MODULE.SCENE_SPECS = {"fixture": {
                "sampling": {"sources": ("sample.log",), "elapsedSources": ("sample.log",), "deviceSources": ("sample.log",), "elapsed": ("elapsedMs", 0.001, "text"), "device": "sampler", "outputs": ("output.bin",)},
                "relation": {"sources": ("relation.jsonl",), "elapsedSources": ("relation.jsonl",), "elapsed": ("elapsedSeconds", 1, "json"), "successOnly": True},
                "training": {"sources": ("history_a.json", "history_b.json", "train.jsonl"), "elapsedSources": ("history_a.json", "history_b.json"), "deviceSources": ("train.jsonl",), "vramSources": ("train.jsonl",), "elapsed": ("elapsedSeconds", 1, "json"), "lastPerSource": True, "deviceField": "device", "vramField": "cudaPeakMemoryAllocatedMiB", "outputs": ("output.bin",)},
            }}
            try:
                rows = MODULE.collect_cost_rows(root, ["fixture"])
            finally:
                MODULE.SCENE_SPECS = original
            by_stage = {row["stage"]: row for row in rows}
            self.assertEqual(by_stage["sampling"]["timeSeconds"], 3.0)
            self.assertEqual(by_stage["sampling"]["device"], "vendor=fixture; renderer=GPU")
            self.assertEqual(by_stage["relation"]["timeSeconds"], 7.0)
            self.assertEqual(by_stage["training"]["timeSeconds"], 9.5)
            self.assertEqual(by_stage["training"]["peakVramMiB"], 12.5)
            self.assertEqual(by_stage["training"]["outputBytes"], 5)
            self.assertEqual(by_stage["training"]["peakCpuRamMiB"], MODULE.UNAVAILABLE)

    def test_missing_values_are_literal_unavailable_in_both_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "preprocessing"
            summary = MODULE.write_cost_bundle(root, output, scenes=["hkust-v3"])
            self.assertEqual(summary["rowCount"], 6)
            payload = json.loads((output / "cost.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["rowStatusCounts"]["unavailable"], 6)
            with (output / "cost.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(all(row["timeSeconds"] == MODULE.UNAVAILABLE for row in rows))


if __name__ == "__main__":
    unittest.main()
