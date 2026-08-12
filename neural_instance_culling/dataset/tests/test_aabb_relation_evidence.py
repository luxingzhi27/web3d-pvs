from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset"
MODEL = ROOT / "model"
if str(DATASET) not in sys.path:
    sys.path.insert(0, str(DATASET))
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

from build_aabb_relation_evidence import _aabb_pose_rows, _rect_overlap_ratio  # noqa: E402


class AABBRelationEvidenceTest(unittest.TestCase):
    def test_overlap_ratio_is_bounded_and_symmetric_for_equal_rectangles(self) -> None:
        rect = np.asarray([[-0.5, -0.5, 0.5, 0.5]], dtype=np.float32)
        ratio = _rect_overlap_ratio(rect, rect)
        self.assertTrue(np.allclose(ratio, 1.0))
        self.assertTrue(np.all((ratio >= 0.0) & (ratio <= 1.0)))

    def test_relation_rows_use_native_candidates_and_front_to_back_depth(self) -> None:
        class Dataset:
            poses = np.zeros(1, dtype=[("camera_world", "<f4", (3,))])
            poses[0]["camera_world"] = [0.0, 0.0, 0.0]

            def visible_slice(self, _pose_index: int) -> tuple[np.ndarray, np.ndarray]:
                return np.asarray([0], dtype=np.uint32), np.asarray([1.0], dtype=np.float32)

            def frustum_slice(self, _pose_index: int) -> np.ndarray:
                return np.asarray([0, 1], dtype=np.uint32)

            def mvp_slice(self, _pose_index: int) -> np.ndarray:
                # x/z, y/z projection with clip w=z.  Instance 0 is in front.
                matrix = np.zeros(16, dtype=np.float32)
                matrix[0] = 1.0
                matrix[5] = 1.0
                matrix[11] = 1.0
                return matrix

        aabbs = np.asarray([
            [-0.2, -0.2, 1.0, 0.2, 0.2, 1.2],
            [-0.3, -0.3, 2.0, 0.3, 0.3, 2.2],
        ], dtype=np.float32)
        centers = (aabbs[:, :3] + aabbs[:, 3:]) * 0.5
        rows, stats = _aabb_pose_rows(
            Dataset(), 0, aabbs, centers, 2,
            min_overlap=0.01,
            min_evidence=0.001,
            min_depth_gap=0.01,
            min_source_area=1e-6,
            min_target_area=1e-7,
            target_chunk_size=8,
            max_targets=0,
            max_sources=0,
            width=320,
            height=180,
        )
        self.assertGreater(rows.shape[0], 0)
        self.assertTrue(np.all(rows[:, 0] == 1.0))
        self.assertTrue(np.all(rows[:, 1] == 0.0))
        self.assertTrue(np.all(rows[:, 2] >= 0.0) and np.all(rows[:, 2] < 12.0))
        self.assertTrue(np.all(rows[:, 3] >= 0.0) and np.all(rows[:, 3] < 3.0))
        self.assertTrue(np.all(rows[:, 5] > 0.0))
        self.assertEqual(stats["candidateCount"], 2)
        self.assertEqual(stats["sourceCount"], 1)


if __name__ == "__main__":
    unittest.main()
