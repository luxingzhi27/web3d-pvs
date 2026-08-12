from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from merge_triangle_depth_layer_caches import CACHE_SCHEMA, CACHE_SCHEMA_V2, EVIDENCE_SCHEMA, merge_caches


class MergeTriangleDepthCacheTest(unittest.TestCase):
    def _write_shard(self, root: Path, name: str, poses: list[int], width: int = 3) -> Path:
        shard = root / name
        shard.mkdir()
        shape = (len(poses), 2, 2, width)
        np.arange(np.prod(shape), dtype="<u4").reshape(shape).tofile(shard / "ids.bin")
        np.arange(np.prod(shape), dtype="<f4").reshape(shape).tofile(shard / "depth.bin")
        np.asarray(poses, dtype="<u4").tofile(shard / "poses.bin")
        (shard / "layer_cache_meta.json").write_text(json.dumps({
            "schema": CACHE_SCHEMA, "modelInputFovYDeg": 66, "width": width, "height": 2,
            "maxLayers": 2, "poseCount": len(poses), "gpuGate": {"hardware": True},
            "files": {"poseIndices": "poses.bin", "instanceIds": "ids.bin", "linearDepth": "depth.bin"},
        }), encoding="utf-8")
        (shard / "run.gpu_evidence.json").write_text(json.dumps({
            "schema": EVIDENCE_SCHEMA,
            "formalReady": True,
            "gpuGate": {"hardware": True},
            "hostGpuBefore": {"nvidiaSmi": {"available": True}, "nvidiaSmiPmon": {"available": True}},
            "hostGpuDuring": {"nvidiaSmi": {"available": True}, "nvidiaSmiPmon": {"available": True}},
        }), encoding="utf-8")
        return shard

    def test_merges_sorted_unique_poses(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = self._write_shard(root, "shard_0", [4, 0])
            second = self._write_shard(root, "shard_1", [3, 1, 2])
            output = root / "merged"
            meta = merge_caches([first, second], output, np.arange(5, dtype=np.int64))
            self.assertEqual(meta["poseCount"], 5)
            self.assertTrue(np.array_equal(np.fromfile(output / meta["files"]["poseIndices"], dtype="<u4"), np.arange(5, dtype="<u4")))

    def test_rejects_duplicate_pose(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = self._write_shard(root, "shard_0", [0])
            second = self._write_shard(root, "shard_1", [0])
            with self.assertRaises(ValueError, msg="duplicate pose must be rejected"):
                merge_caches([first, second], root / "merged", np.asarray([0], dtype=np.int64))

    def test_rejects_non_hardware_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shard = self._write_shard(root, "shard_0", [0])
            meta_path = shard / "layer_cache_meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["gpuGate"]["hardware"] = False
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            with self.assertRaises(ValueError, msg="software cache must not enter formal merge"):
                merge_caches([shard], root / "merged", np.asarray([0], dtype=np.int64))

    def test_v2_keeps_representative_source_pose_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shards = []
            for shard_number, render_ids in enumerate(([2, 0], [3, 1])):
                shard = root / f"v2_shard_{shard_number}"
                shard.mkdir()
                shape = (len(render_ids), 2, 2, 2)
                np.arange(np.prod(shape), dtype="<u4").reshape(shape).tofile(shard / "ids.bin")
                np.arange(np.prod(shape), dtype="<f4").reshape(shape).tofile(shard / "depth.bin")
                np.asarray(render_ids, dtype="<u4").tofile(shard / "render.bin")
                np.asarray([7, 7] if shard_number == 0 else [8, 8], dtype="<u4").tofile(shard / "source.bin")
                (shard / "layer_cache_meta.json").write_text(json.dumps({
                    "schema": CACHE_SCHEMA_V2, "modelInputFovYDeg": 66, "width": 2, "height": 2,
                    "maxLayers": 2, "poseCount": len(render_ids), "gpuGate": {"hardware": True},
                    "firstLayerReference": {"checks": len(render_ids), "mismatches": 0},
                    "poseIdSemantics": "renderPoseId/sourcePoseIndex",
                    "files": {"poseIndices": "render.bin", "renderPoseIds": "render.bin", "sourcePoseIndices": "source.bin", "instanceIds": "ids.bin", "linearDepth": "depth.bin"},
                }), encoding="utf-8")
                (shard / "run.gpu_evidence.json").write_text(json.dumps({
                    "schema": EVIDENCE_SCHEMA, "formalReady": True, "gpuGate": {"hardware": True},
                    "hostGpuBefore": {"nvidiaSmi": {"available": True}, "nvidiaSmiPmon": {"available": True}},
                    "hostGpuDuring": {"nvidiaSmi": {"available": True}, "nvidiaSmiPmon": {"available": True}},
                }), encoding="utf-8")
                shards.append(shard)
            meta = merge_caches(shards, root / "merged", np.asarray([0, 1, 2, 3], dtype=np.int64))
            self.assertEqual(meta["schema"], CACHE_SCHEMA_V2)
            self.assertTrue(np.array_equal(
                np.fromfile(root / "merged" / meta["files"]["sourcePoseIndices"], dtype="<u4"),
                np.asarray([7, 8, 7, 8], dtype="<u4"),
            ))


if __name__ == "__main__":
    unittest.main()
