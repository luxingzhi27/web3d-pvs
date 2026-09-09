#!/usr/bin/env python3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_hzb_timing_pose_plan import POSE_DTYPE, build_plan  # noqa: E402


class BuildHzbTimingPosePlanTest(unittest.TestCase):
    def make_dataset(self, root: Path) -> Path:
        dataset = root / "dataset"
        dataset.mkdir()
        pose_count = 12
        poses = np.zeros((pose_count,), dtype=POSE_DTYPE)
        poses["split"] = 3
        poses["camera_view"] = np.asarray([[1.0 + index / 12.0, 1.0] for index in range(pose_count)], dtype=np.float32)
        poses.tofile(dataset / "poses.bin")
        counts = np.arange(1, pose_count + 1, dtype=np.uint64)
        offsets = np.concatenate((np.asarray([0], dtype=np.uint64), np.cumsum(counts, dtype=np.uint64)))
        offsets.tofile(dataset / "candidate_offsets.bin")
        np.arange(int(offsets[-1]), dtype=np.uint32).tofile(dataset / "candidate_ids.bin")
        (dataset / "dataset_meta.json").write_text(
            '{"poseCount": 12, "poseStrideBytes": 64, "splitIds": {"test": 3}, '
            '"candidateSemantics": "synthetic"}\n',
            encoding="utf-8",
        )
        return dataset

    def test_selects_each_candidate_stratum_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self.make_dataset(root)
            first = build_plan(dataset, root / "first.json", pose_count=6, strata_count=3)
            second = build_plan(dataset, root / "second.json", pose_count=6, strata_count=3)
            self.assertEqual(first["poseIndices"], second["poseIndices"])
            self.assertEqual(first["targetPoseCount"], 6)
            self.assertEqual([item["selectedPoseCount"] for item in first["strata"]], [2, 2, 2])
            self.assertEqual(len(set(first["poseIndices"])), 6)
            self.assertEqual(first["poseIndices"], [0, 3, 4, 7, 8, 11])
            self.assertTrue(all(item["aspect"] is not None for item in first["poses"]))
            self.assertEqual(first["source"]["splitPoseCount"], 12)

    def test_rejects_a_plan_larger_than_the_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self.make_dataset(root)
            with self.assertRaisesRegex(ValueError, "cannot select"):
                build_plan(dataset, root / "plan.json", pose_count=13)


if __name__ == "__main__":
    unittest.main()
