from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_DIR))

import run_m5_visual_safety_image_evaluation as m5_image  # noqa: E402
import run_m5_component_image_batch as m5_batch  # noqa: E402
import evaluate_viewcell_image_per as viewcell_image  # noqa: E402


class M5VisualSafetyImageEvaluationTests(unittest.TestCase):
    def test_default_experiment_template_remains_registered(self) -> None:
        self.assertEqual(
            m5_image.experiment_name("m5_visual_mass_soft", 20260803),
            "m5_visual_mass_soft_rvl_strong_v2_hkust_spatial_fov66_seed20260803_full40",
        )

    def test_new_matrix_can_resolve_an_independent_experiment_template(self) -> None:
        self.assertEqual(
            m5_image.experiment_name(
                "pvs_m5_subpose_robust_v1_hkust_spatial_fov66",
                20260801,
                "{variant}_seed{seed}_full40",
            ),
            "pvs_m5_subpose_robust_v1_hkust_spatial_fov66_seed20260801_full40",
        )

    def test_zero_subposes_selects_all_dense_subposes(self) -> None:
        selected = viewcell_image.ViewcellDataset.select_subpose_indices(10, 43, 0)
        self.assertEqual(selected.tolist(), list(range(10, 43)))

    def test_positive_subpose_selection_is_deterministic_and_evenly_spaced(self) -> None:
        selected = viewcell_image.ViewcellDataset.select_subpose_indices(10, 43, 4)
        self.assertEqual(selected.tolist(), [10, 20, 31, 42])

    def test_negative_subpose_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            viewcell_image.ViewcellDataset.select_subpose_indices(10, 43, -1)

    def test_gpu_snapshot_writes_both_host_observations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(
                viewcell_image.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=0, stdout="gpu evidence\n", stderr=""),
            ) as run:
                evidence = viewcell_image._capture_gpu_snapshot(Path(temp_dir), "during")
            self.assertTrue(evidence["complete"])
            self.assertEqual(run.call_count, 2)
            self.assertTrue((Path(temp_dir) / "nvidia_smi_during.txt").is_file())
            self.assertTrue((Path(temp_dir) / "nvidia_smi_pmon_during.txt").is_file())

    def test_image_threshold_rejects_frozen_workpoint_below_pose_recall_floor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = Path(temp_dir) / "calibration_ready_summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "protocol": "calibration_ready_pre_test",
                        "testEvaluationCount": 0,
                        "frozenThreshold": 0.2,
                        "calibration": {"selected": {"threshold": 0.2, "pose_recall": 0.94}},
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                threshold=None,
                threshold_policy="weighted_precision",
                target_weighted_recall=0.99,
                minimum_pose_recall=0.95,
            )
            with self.assertRaisesRegex(RuntimeError, "below required"):
                viewcell_image.resolve_threshold(
                    args,
                    {"eval_summary": str(summary)},
                    SimpleNamespace(threshold=0.0),
                )

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

    def test_chunk_manifest_size_limit_reduces_sample_window(self) -> None:
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
            "batches": [{"batchId": "a", "samples": []}, {"batchId": "b", "samples": []}],
        }
        for batch in base["batches"]:
            for index in range(4):
                sample = m5_batch.minimal_sample(f"{batch['batchId']}-{index}")
                sample["testPadding"] = "x" * 5000
                batch["samples"].append(sample)

        with tempfile.TemporaryDirectory() as temp_dir:
            chunks = m5_batch.build_chunk_manifests(base, Path(temp_dir), 4, 20_000)
            self.assertEqual(len(chunks), 4)
            for path, _batch_map in chunks:
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(payload["chunking"]["sampleLimitPerSourceBatch"], 1)
                self.assertLessEqual(path.stat().st_size, 20_000)


if __name__ == "__main__":
    unittest.main()
