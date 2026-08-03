from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_DIR))

import run_m5_visual_safety_image_evaluation as m5_image  # noqa: E402
import run_m5_component_image_batch as m5_batch  # noqa: E402


class M5VisualSafetyImageEvaluationTests(unittest.TestCase):
    def test_percentile_uses_linear_interpolation(self) -> None:
        self.assertAlmostEqual(m5_image.percentile([0.0, 0.1, 0.2, 0.3], 0.95), 0.285)

    def test_load_viewcell_metrics_groups_batches_and_computes_p95(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            render_dir = output / "true_glb_render"
            render_dir.mkdir()
            rows = []
            for batch_id, values in {"validation": [0.0, 0.1, 0.2], "calibration": [0.3, 0.4]}.items():
                for index, miss in enumerate(values):
                    rows.append(
                        {
                            "batchId": batch_id,
                            "imageMetrics": {
                                "missPixelRate": miss,
                                "wrongInstancePixelRate": miss / 2.0,
                                "PER": miss * 2.0,
                            },
                        }
                    )
            (render_dir / "sample_image_metrics.json").write_text(json.dumps(rows), encoding="utf-8")

            result = m5_image.load_viewcell_metrics(output, ["validation", "calibration"])

            self.assertEqual(result["validation"]["sampleCount"], 3)
            self.assertAlmostEqual(result["validation"]["viewCellMissPixelRateMean"], 0.1)
            self.assertAlmostEqual(result["validation"]["viewCellMissPixelRateP95"], 0.19)
            self.assertAlmostEqual(result["calibration"]["viewCellMissPixelRateMax"], 0.4)

    def test_load_viewcell_metrics_rejects_missing_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            render_dir = Path(temp_dir) / "true_glb_render"
            render_dir.mkdir()
            (render_dir / "sample_image_metrics.json").write_text(
                json.dumps(
                    [
                        {
                            "batchId": "validation",
                            "imageMetrics": {
                                "missPixelRate": 0.0,
                                "wrongInstancePixelRate": 0.0,
                                "PER": 0.0,
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                m5_image.load_viewcell_metrics(Path(temp_dir), ["validation", "calibration"])

    def test_chunk_manifest_preserves_inventory_and_sample_partition(self) -> None:
        binding = {
            "schema": m5_batch.INSTANCE_BINDING_SCHEMA,
            "byGlobalGlbId": {
                "0": {
                    "globalGlbId": 0,
                    "componentGlobalIds": [0],
                    "instanceCount": 1,
                    "renderable": True,
                }
            },
            "componentToBinding": {
                "0": {"componentGlobalId": 0, "globalGlbId": 0, "instanceIndex": 0}
            },
        }
        base = {
            "schema": m5_batch.INSTANCE_RENDER_BATCH_MANIFEST_SCHEMA,
            "idEncoding": m5_batch.INSTANCE_ID_ENCODING,
            "renderFovYDeg": m5_batch.RENDER_FOV_Y_DEG,
            "modelInputFovYDeg": m5_batch.MODEL_INPUT_FOV_Y_DEG,
            "formalImageEvaluationReady": False,
            "selectedGlbs": [0],
            "reference": {"mode": "full_scene_renderable_instances", "idSource": "componentGlobalId"},
            "prediction": {"field": "predictionComponentIds"},
            "instanceBindings": binding,
            "glbAabbs": {"0": {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]}},
            "componentAabbs": {"0": {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]}},
            "spatialCulling": {"schema": "aabb-frustum-conservative-v1"},
            "batches": [
                {"batchId": "a", "samples": [m5_batch.minimal_sample(f"a-{i}") for i in range(3)]},
                {"batchId": "b", "samples": [m5_batch.minimal_sample(f"b-{i}") for i in range(3)]},
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            chunks = m5_batch.build_chunk_manifests(base, Path(temp_dir), 2)
            self.assertEqual(len(chunks), 2)
            for chunk_index, (path, batch_map) in enumerate(chunks):
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(payload["selectedGlbs"], [0])
                self.assertEqual(payload["chunking"]["chunkNumber"], chunk_index)
                self.assertTrue(payload["chunking"]["completeInventoryRetained"])
                self.assertEqual(sum(len(batch["samples"]) for batch in payload["batches"]), 4 if chunk_index == 0 else 2)
                self.assertEqual(set(batch_map.values()), {"a", "b"})


if __name__ == "__main__":
    unittest.main()
