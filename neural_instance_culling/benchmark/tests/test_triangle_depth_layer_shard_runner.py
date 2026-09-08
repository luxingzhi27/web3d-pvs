from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from neural_instance_culling.benchmark.run_triangle_depth_layer_shards import (
    command_for_manifest,
    completed_formal_cache,
    completed_sparse_cache,
    remove_dense_payload,
)


class TriangleDepthLayerShardRunnerTests(unittest.TestCase):
    def test_completed_cache_requires_formal_hardware_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "ids.bin").write_bytes(b"ids")
            (output / "depth.bin").write_bytes(b"depth")
            (output / "layer_cache_meta.json").write_text(
                json.dumps({
                    "gpuGate": {"hardware": True},
                    "files": {"instanceIds": "ids.bin", "linearDepth": "depth.bin"},
                }),
                encoding="utf-8",
            )
            evidence = {
                "schema": "triangle-depth-layer-gpu-evidence-v1",
                "formalReady": True,
                "gpuGate": {"hardware": True},
            }
            (output / "triangle_depth.bin.gpu_evidence.json").write_text(
                json.dumps(evidence), encoding="utf-8"
            )
            self.assertTrue(completed_formal_cache(output))
            evidence["formalReady"] = False
            (output / "triangle_depth.bin.gpu_evidence.json").write_text(
                json.dumps(evidence), encoding="utf-8"
            )
            self.assertFalse(completed_formal_cache(output))

    def test_dense_payload_is_removed_only_after_sparse_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "ids.bin").write_bytes(b"ids")
            (output / "depth.bin").write_bytes(b"depth")
            (output / "layer_cache_meta.json").write_text(
                json.dumps({
                    "files": {"instanceIds": "ids.bin", "linearDepth": "depth.bin"},
                }),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                remove_dense_payload(output)

            sparse = output / "sparse"
            sparse.mkdir()
            files = {
                "relationMoments": "relation.bin",
                "relationDepthOffsets": "depth_offsets.bin",
                "relationDepthValues": "depth_values.bin",
                "survivalObservations": "survival.bin",
                "renderPoseIds": "render.bin",
                "sourcePoseIndices": "source.bin",
            }
            for name in files.values():
                (sparse / name).write_bytes(b"x")
            (sparse / "sparse_relation_meta.json").write_text(
                json.dumps({
                    "schema": "triangle-depth-relation-sparse-shard-v1",
                    "formalReady": True,
                    "trainOnly": True,
                    "poseCount": 1,
                    "files": files,
                }),
                encoding="utf-8",
            )
            self.assertTrue(completed_sparse_cache(output))
            self.assertEqual(remove_dense_payload(output), len(b"ids") + len(b"depth"))
            self.assertFalse((output / "ids.bin").exists())
            self.assertFalse((output / "depth.bin").exists())

    def test_command_registers_timeout_hardware_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "out"
            output.mkdir()
            (output / "triangle_depth.bin.pose_indices.partial").write_bytes(b"\0\0\0\0")
            command = command_for_manifest(
                root / "shard_00.json",
                output,
                Path("/usr/bin/google-chrome"),
                320,
                180,
                6,
                3_600_000,
                True,
                True,
            )
            self.assertIn("--require-hardware-gpu", command)
            self.assertIn("--resume", command)
            self.assertEqual(command[command.index("--timeout-ms") + 1], "3600000")


if __name__ == "__main__":
    unittest.main()
