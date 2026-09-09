from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

BENCHMARK = Path(__file__).resolve().parents[1]
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))

from build_gt_convergence_pose_plan import nested_offsets, select_stratified_poses  # noqa: E402


class GtConvergencePosePlanTests(unittest.TestCase):
    def test_disk_sequence_is_nested_and_stays_inside_radius(self) -> None:
        first = nested_offsets(
            16,
            shape="horizontal_disk",
            radius=2.0,
            half_up=0.0,
            forward=np.asarray([0.0, 0.0, -1.0]),
        )
        second = nested_offsets(
            128,
            shape="horizontal_disk",
            radius=2.0,
            half_up=0.0,
            forward=np.asarray([0.0, 0.0, -1.0]),
        )
        np.testing.assert_allclose(first, second[:16])
        np.testing.assert_allclose(second[0], 0.0)
        self.assertLessEqual(float(np.linalg.norm(second[:, [0, 2]], axis=1).max()), 2.0)
        np.testing.assert_allclose(second[:, 1], 0.0)

    def test_camera_box_sequence_is_nested_and_bounded(self) -> None:
        offsets = nested_offsets(
            128,
            shape="camera_aligned_box",
            radius=2.5,
            half_up=1.0,
            forward=np.asarray([0.0, 0.0, -1.0]),
        )
        np.testing.assert_allclose(offsets[0], 0.0)
        self.assertLessEqual(float(np.abs(offsets[:, 0]).max()), 2.5)
        self.assertLessEqual(float(np.abs(offsets[:, 1]).max()), 1.0)
        self.assertLessEqual(float(np.abs(offsets[:, 2]).max()), 2.5)

    def test_stratified_selection_is_deterministic_and_validation_local(self) -> None:
        pose_indices = np.arange(10, 50, dtype=np.int64)
        categories = np.arange(60, dtype=np.uint8) % 4
        angles = np.linspace(-np.pi, np.pi, 60, endpoint=False)
        forwards = np.stack((-np.sin(angles), np.zeros(60), -np.cos(angles)), axis=1)
        counts = np.arange(60, dtype=np.int64)
        selected = select_stratified_poses(
            pose_indices,
            categories,
            forwards,
            counts * 10,
            counts,
            counts.astype(np.float64) * 100.0,
            count=12,
            seed=7,
        )
        repeated = select_stratified_poses(
            pose_indices,
            categories,
            forwards,
            counts * 10,
            counts,
            counts.astype(np.float64) * 100.0,
            count=12,
            seed=7,
        )
        self.assertEqual(selected.size, 12)
        self.assertTrue(np.all(np.isin(selected, pose_indices)))
        self.assertEqual(set(categories[selected].tolist()), {0, 1, 2, 3})
        np.testing.assert_array_equal(selected, repeated)


if __name__ == "__main__":
    unittest.main()
