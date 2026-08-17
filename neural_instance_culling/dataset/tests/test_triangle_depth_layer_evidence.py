from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset"
MODEL = ROOT / "model"
if str(DATASET) not in sys.path: sys.path.insert(0, str(DATASET))
if str(MODEL) not in sys.path: sys.path.insert(0, str(MODEL))

from build_triangle_depth_layer_evidence import (  # noqa: E402
    _adjacent_occlusion_pixel_counts,
    build_evidence,
    load_layer_cache,
    metric_ray_depths_for_cache_row,
)
from pose_csr_dataset import DIRECTIONAL_POSE_DTYPE  # noqa: E402


class TriangleDepthEvidenceTest(unittest.TestCase):
    def test_same_component_depth_repeat_is_not_an_occlusion_event(self) -> None:
        layers = np.asarray([
            [[1, 1], [2, 2]],
            [[1, 3], [2, 0]],
            [[4, 3], [2, 0]],
        ], dtype=np.uint32)
        counts = _adjacent_occlusion_pixel_counts(layers, {1, 2, 3, 4})
        self.assertEqual(counts, {3: 1, 4: 1})

    def test_first_layer_and_adjacent_relation_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset_dir = root / "dataset"
            cache_dir = root / "cache"
            output_dir = root / "evidence"
            dataset_dir.mkdir(); cache_dir.mkdir()
            runtime_meta = {
                "sceneBounds": {"min": [0, 0, 0], "max": [10, 10, 10]},
                "componentRecords": [
                    {"componentGlobalId": 0, "globalGlbId": 0, "bounds": {"min": [0, 0, 1], "max": [1, 1, 2]}},
                    {"componentGlobalId": 1, "globalGlbId": 1, "bounds": {"min": [0, 0, 3], "max": [1, 1, 4]}},
                    {"componentGlobalId": 2, "globalGlbId": 1, "bounds": {"min": [4, 0, 3], "max": [5, 1, 4]}},
                ],
            }
            runtime_path = root / "runtime.json"
            runtime_path.write_text(json.dumps(runtime_meta), encoding="utf-8")
            poses = np.zeros(1, dtype=DIRECTIONAL_POSE_DTYPE)
            poses[0]["camera_world"] = [0, 0, 0]
            poses[0]["camera_forward"] = [0, 0, 1]
            poses[0]["camera_view"] = [1.0, 1.0]
            poses[0]["split"] = 0
            poses.tofile(dataset_dir / "poses.bin")
            np.asarray([0, 1], dtype="<u8").tofile(dataset_dir / "visible_offsets.bin")
            np.asarray([1], dtype="<u4").tofile(dataset_dir / "visible_ids.bin")
            np.asarray([1.0], dtype="<f4").tofile(dataset_dir / "visible_weights.bin")
            np.asarray([0, 3], dtype="<u8").tofile(dataset_dir / "frustum_offsets.bin")
            np.asarray([0, 1, 2], dtype="<u4").tofile(dataset_dir / "frustum_ids.bin")
            (dataset_dir / "dataset_meta.json").write_text(json.dumps({"splitIds": {"train": 0, "validation": 1, "calibration": 2, "test": 3}, "poseStrideBytes": 64}), encoding="utf-8")
            ids = np.zeros((1, 2, 2, 2), dtype="<u4")
            depths = np.ones((1, 2, 2, 2), dtype="<f4")
            ids[0, 0] = 1
            ids[0, 1] = np.asarray([[2, 2], [2, 0]], dtype="<u4")
            depths[0, 0] = 0.2
            depths[0, 1] = np.asarray([[0.6, 0.6], [0.6, 1.0]], dtype="<f4")
            ids.tofile(cache_dir / "ids.bin")
            depths.tofile(cache_dir / "depth.bin")
            np.asarray([0], dtype="<u4").tofile(cache_dir / "poses.bin")
            (cache_dir / "layer_cache_meta.json").write_text(json.dumps({
                "schema": "triangle-depth-layer-cache-v1", "poseCount": 1, "maxLayers": 2, "width": 2, "height": 2, "modelInputFovYDeg": 66, "cameraFarMeters": 10.0,
                "files": {"poseIndices": "poses.bin", "instanceIds": "ids.bin", "linearDepth": "depth.bin"},
            }), encoding="utf-8")
            (cache_dir / "triangle_depth.gpu_evidence.json").write_text(json.dumps({
                "schema": "triangle-depth-layer-gpu-evidence-v1",
                "formalReady": True,
                "gpuGate": {"hardware": True},
                "hostGpuBefore": {
                    "nvidiaSmi": {"available": True},
                    "nvidiaSmiPmon": {"available": True},
                },
                "hostGpuDuring": {
                    "nvidiaSmi": {"available": True},
                    "nvidiaSmiPmon": {"available": True},
                },
            }), encoding="utf-8")
            args = type("Args", (), {
                "dataset_dir": str(dataset_dir), "runtime_meta": str(runtime_path), "layer_cache_dir": str(cache_dir), "output_dir": str(output_dir),
                "splits": "train", "source_k": 2, "min_depth_gap": 0.01, "max_poses": 0, "allow_outside_candidate": False,
            })()
            meta = build_evidence(args)
            self.assertEqual(meta["schema"], "triangle-depth-layer-evidence-v3")
            self.assertTrue(meta["pixelWeightPolicy"].endswith("sum_to_one"))
            self.assertGreater(meta["stats"]["layerPairCount"], 0)
            self.assertGreater(meta["stats"]["survivalObservationCount"], 0)
            source_ids = np.fromfile(output_dir / "source_ids_uint32.bin", dtype="<u4")
            self.assertEqual(source_ids.size, 3 * 12 * 3 * 2)

    def test_rejects_cache_without_formal_gpu_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache = root / "cache"
            cache.mkdir()
            np.zeros(1, dtype="<u4").tofile(cache / "poses.bin")
            np.zeros((1, 2, 1, 1), dtype="<u4").tofile(cache / "ids.bin")
            np.zeros((1, 2, 1, 1), dtype="<f4").tofile(cache / "depth.bin")
            (cache / "layer_cache_meta.json").write_text(json.dumps({
                "schema": "triangle-depth-layer-cache-v1", "poseCount": 1, "maxLayers": 2,
                "width": 1, "height": 1, "modelInputFovYDeg": 66,
                "files": {"poseIndices": "poses.bin", "instanceIds": "ids.bin", "linearDepth": "depth.bin"},
            }), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_layer_cache(cache)

    def test_v2_cache_uses_subpose_camera_and_allows_shared_source_pose(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset_dir = root / "dataset"
            cache_dir = root / "cache"
            output_dir = root / "evidence_v2"
            dataset_dir.mkdir(); cache_dir.mkdir()
            runtime_meta = {
                "sceneBounds": {"min": [0, 0, 0], "max": [10, 10, 10]},
                "componentRecords": [
                    {"componentGlobalId": 0, "globalGlbId": 0, "bounds": {"min": [0, 0, 1], "max": [1, 1, 2]}},
                    {"componentGlobalId": 1, "globalGlbId": 1, "bounds": {"min": [0, 0, 3], "max": [1, 1, 4]}},
                    {"componentGlobalId": 2, "globalGlbId": 2, "bounds": {"min": [0, 0, 5], "max": [1, 1, 6]}},
                ],
            }
            runtime_path = root / "runtime.json"
            runtime_path.write_text(json.dumps(runtime_meta), encoding="utf-8")
            poses = np.zeros(1, dtype=DIRECTIONAL_POSE_DTYPE)
            poses[0]["camera_world"] = [0, 0, 0]
            poses[0]["camera_forward"] = [0, 0, 1]
            poses[0]["camera_view"] = [1.0, 1.0]
            poses[0]["split"] = 0
            poses.tofile(dataset_dir / "poses.bin")
            np.asarray([0, 1], dtype="<u8").tofile(dataset_dir / "visible_offsets.bin")
            np.asarray([1], dtype="<u4").tofile(dataset_dir / "visible_ids.bin")
            np.asarray([1.0], dtype="<f4").tofile(dataset_dir / "visible_weights.bin")
            np.asarray([0, 3], dtype="<u8").tofile(dataset_dir / "frustum_offsets.bin")
            np.asarray([0, 1, 2], dtype="<u4").tofile(dataset_dir / "frustum_ids.bin")
            (dataset_dir / "dataset_meta.json").write_text(json.dumps({
                "splitIds": {"train": 0, "validation": 1, "calibration": 2, "test": 3},
                "poseStrideBytes": 64,
            }), encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "schema": "triangle-depth-layer-render-manifest-v2",
                "cameraFar": 10.0,
                "poses": [
                    {"renderPoseId": 0, "sourcePoseIndex": 0, "cameraWorld": [0, 0, 0], "cameraForward": [0, 0, 1], "cameraView": [0, 0, 1, 1, 1]},
                    {"renderPoseId": 1, "sourcePoseIndex": 0, "cameraWorld": [0.5, 0, 0], "cameraForward": [0, 0, 1], "cameraView": [0, 0, 1, 1, 1]},
                ],
            }), encoding="utf-8")
            ids = np.zeros((2, 2, 2, 2), dtype="<u4")
            depths = np.ones((2, 2, 2, 2), dtype="<f4")
            ids[:, 0] = 1
            ids[:, 1] = np.asarray([[2, 2], [2, 0]], dtype="<u4")
            depths[:, 0] = 0.2
            depths[:, 1] = np.asarray([[0.6, 0.6], [0.6, 1.0]], dtype="<f4")
            ids.tofile(cache_dir / "ids.bin")
            depths.tofile(cache_dir / "depth.bin")
            np.asarray([0, 1], dtype="<u4").tofile(cache_dir / "render.bin")
            np.asarray([0, 0], dtype="<u4").tofile(cache_dir / "source.bin")
            (cache_dir / "layer_cache_meta.json").write_text(json.dumps({
                "schema": "triangle-depth-layer-cache-v2", "poseCount": 2, "maxLayers": 2,
                "width": 2, "height": 2, "modelInputFovYDeg": 66,
                "sourceManifest": str(manifest),
                "firstLayerReference": {"checks": 2, "mismatches": 0},
                "files": {"poseIndices": "render.bin", "renderPoseIds": "render.bin", "sourcePoseIndices": "source.bin", "instanceIds": "ids.bin", "linearDepth": "depth.bin"},
            }), encoding="utf-8")
            (cache_dir / "triangle_depth.gpu_evidence.json").write_text(json.dumps({
                "schema": "triangle-depth-layer-gpu-evidence-v1", "formalReady": True,
                "gpuGate": {"hardware": True},
                "hostGpuBefore": {"nvidiaSmi": {"available": True}, "nvidiaSmiPmon": {"available": True}},
                "hostGpuDuring": {"nvidiaSmi": {"available": True}, "nvidiaSmiPmon": {"available": True}},
            }), encoding="utf-8")
            args = type("Args", (), {
                "dataset_dir": str(dataset_dir), "runtime_meta": str(runtime_path), "layer_cache_dir": str(cache_dir), "output_dir": str(output_dir),
                "splits": "train", "source_k": 2, "min_depth_gap": 0.01, "max_poses": 0, "allow_outside_candidate": False,
            })()
            meta = build_evidence(args)
            self.assertEqual(meta["stats"]["renderPoseCount"], 2)
            self.assertEqual(meta["stats"]["sourcePoseCount"], 1)
            self.assertTrue(meta["stats"]["representativeSubposeCache"])

    def test_normalized_axial_cache_decodes_to_metric_ray_range(self) -> None:
        encoded = np.full((1, 1, 2), 0.5, dtype=np.float32)
        decoded = metric_ray_depths_for_cache_row(
            {
                "linearDepthEncoding": "camera_forward_axial_depth_divided_by_camera_far",
                "cameraFarMeters": 10.0,
            },
            encoded,
            np.asarray([0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32),
        )
        expected = 5.0 * np.sqrt(1.0 + 0.5**2)
        self.assertTrue(np.allclose(decoded, expected, rtol=1e-6, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
