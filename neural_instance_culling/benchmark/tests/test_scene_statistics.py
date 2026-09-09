from __future__ import annotations

import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

BENCHMARK = Path(__file__).resolve().parents[1]
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))

from scene_statistics import collect_scene_statistics, write_scene_statistics_csv  # noqa: E402


def _minimal_glb(index_count: int = 6) -> bytes:
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": index_count * 2}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": index_count * 2}],
        "accessors": [{"bufferView": 0, "componentType": 5123, "count": index_count, "type": "SCALAR"}],
        "meshes": [{"primitives": [{"attributes": {}, "indices": 0}]}],
    }
    encoded = json.dumps(document, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((4 - len(encoded) % 4) % 4)
    binary = b"\0" * (index_count * 2)
    total = 12 + 8 + len(encoded) + 8 + len(binary)
    return b"glTF" + struct.pack("<II", 2, total) + struct.pack("<II", len(encoded), 0x4E4F534A) + encoded + struct.pack("<II", len(binary), 0x004E4942) + binary


class SceneStatisticsTests(unittest.TestCase):
    def test_csv_contains_scene_range_reuse_split_and_candidate_gt_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = root / "assets"
            assets.mkdir()
            (assets / "prototype.glb").write_bytes(_minimal_glb())
            (assets / "glbIndex.json").write_text(
                json.dumps({"entries": [{"globalId": 0, "path": "prototype.glb"}]}),
                encoding="utf-8",
            )
            (assets / "runtimeVisibilityMeta.json").write_text(
                json.dumps(
                    {
                        "sceneBounds": {"center": [0, 0, 0], "size": [2, 4, 4]},
                        "componentRecords": [
                            {"componentGlobalId": 0, "globalGlbId": 0, "bounds": {"center": [0, 0, 0], "size": [1, 1, 1]}},
                            {"componentGlobalId": 1, "globalGlbId": 0, "bounds": {"center": [1, 0, 0], "size": [1, 1, 1]}},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "dataset_meta.json").write_text(
                json.dumps(
                    {
                        "splitIds": {"train": 0, "validation": 1, "calibration": 2, "test": 3},
                        "poseStrideBytes": 64,
                    }
                ),
                encoding="utf-8",
            )
            poses = np.zeros((4,), dtype=np.dtype({"names": ["camera_norm", "camera_world", "camera_forward", "camera_view", "split", "category", "reserved0", "reserved1"], "formats": [("<f4", (3,)), ("<f4", (3,)), ("<f4", (3,)), ("<f4", (2,)), "u1", "u1", "<u2", "<u4"], "offsets": [0, 12, 24, 36, 44, 45, 46, 48], "itemsize": 64}))
            poses["split"] = [0, 1, 2, 3]
            poses.tofile(dataset / "poses.bin")
            np.asarray([0, 1, 2, 2, 2], dtype=np.uint64).tofile(dataset / "visible_offsets.bin")
            np.asarray([0, 1], dtype=np.uint32).tofile(dataset / "visible_ids.bin")
            np.asarray([2.0, 3.0], dtype=np.float32).tofile(dataset / "visible_weights.bin")
            np.asarray([0, 1, 3, 3, 4], dtype=np.uint64).tofile(dataset / "candidate_offsets.bin")
            np.asarray([0, 1, 0, 1], dtype=np.uint32).tofile(dataset / "candidate_ids.bin")

            row = collect_scene_statistics(
                "fixture",
                dataset,
                assets / "runtimeVisibilityMeta.json",
                assets / "glbIndex.json",
                assets,
            )
            self.assertEqual(row["instance_count"], 2)
            self.assertEqual(row["glb_count"], 1)
            self.assertEqual(row["prototype_triangle_count"], 2)
            self.assertEqual(row["expanded_triangle_count"], 4)
            self.assertEqual(row["instance_glb_reuse_factor"], 2.0)
            self.assertEqual(row["train_pose_count"], 1)
            self.assertEqual(row["test_zero_gt_pose_count"], 1)
            self.assertGreater(row["scene_diagonal"], 0.0)

            output = root / "scene_statistics.csv"
            write_scene_statistics_csv(output, [row])
            text = output.read_text(encoding="utf-8")
            self.assertIn("prototype_triangle_count", text)
            self.assertIn("test_zero_gt_pose_count", text)


if __name__ == "__main__":
    unittest.main()
