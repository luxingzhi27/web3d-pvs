from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from neural_instance_culling.model.v5.training_data import (
    V5SceneTrainingData,
    _camera_basis,
    _query_geometry_numpy,
    _support_points,
)


class _DatasetFixture:
    def __init__(self) -> None:
        self.candidate_counts = np.asarray([4, 0, 2], dtype=np.int64)

    def visible_slice(self, pose_id: int):
        rows = {
            0: (np.asarray([0], dtype=np.uint32), np.asarray([2.0], dtype=np.float32)),
            1: (np.zeros((0,), dtype=np.uint32), np.zeros((0,), dtype=np.float32)),
            2: (np.asarray([1], dtype=np.uint32), np.asarray([3.0], dtype=np.float32)),
        }
        return rows[int(pose_id)]


class TrainingDataContractTests(unittest.TestCase):
    def _scene(self) -> V5SceneTrainingData:
        scene = V5SceneTrainingData.__new__(V5SceneTrainingData)
        scene.scene_id = "fixture"
        scene.dataset = _DatasetFixture()
        scene.train_split = SimpleNamespace(pose_indices=np.asarray([0, 1, 2], dtype=np.int64))
        scene.degenerate_unit_ids = np.zeros((0,), dtype=np.int64)
        return scene

    def test_denominators_record_the_actual_pose_sampling_population(self) -> None:
        denominators = self._scene()._compute_train_denominators()
        self.assertEqual(denominators.pose_count, 3)
        self.assertEqual(denominators.eligible_pose_count, 2)
        self.assertEqual(denominators.visible_occurrences, 2)
        self.assertEqual(denominators.visible_weight_sum, 5.0)

    def test_visible_degenerate_unit_is_rejected(self) -> None:
        scene = self._scene()
        scene.degenerate_unit_ids = np.asarray([1], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "visible degenerate"):
            scene._compute_train_denominators()


class TrainingGeometryTest(unittest.TestCase):
    def test_disk_and_box_supports_follow_registered_geometry(self) -> None:
        basis = _camera_basis(np.asarray([0.0, 0.0, -1.0], dtype=np.float32))
        disk = _support_points(
            np.zeros(3), basis, "horizontal_disk", np.asarray([2.0, 2.0, 0.0])
        )
        box = _support_points(
            np.zeros(3), basis, "camera_aligned_box", np.asarray([2.5, 2.5, 1.0])
        )
        self.assertEqual(disk.shape, (9, 3))
        self.assertTrue(np.allclose(disk[:, 1], 0.0))
        self.assertTrue(np.allclose(np.linalg.norm(disk[1:, [0, 2]], axis=1), 2.0))
        self.assertEqual(np.unique(box[1:], axis=0).shape[0], 8)

    def test_query_geometry_uses_exact_16_channel_layout(self) -> None:
        basis = _camera_basis(np.asarray([0.0, 0.0, -1.0], dtype=np.float32))
        query = _query_geometry_numpy(
            np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            np.asarray([0.0, 0.0, 3.0], dtype=np.float32),
            basis,
            np.asarray([2.0, 0.0, 2.0], dtype=np.float32),
            np.asarray([0.8, 0.6], dtype=np.float32),
            0.0,
            0.01,
            1000.0,
        )
        self.assertEqual(query.shape, (1, 16))
        self.assertTrue(np.allclose(query[0, :3], [0.0, 0.0, 1.0]))
        self.assertAlmostEqual(float(query[0, 6]), float(np.log(4.0)), places=6)
        self.assertAlmostEqual(float(query[0, 7]), 0.25, places=6)
        self.assertEqual(float(query[0, 13]), 0.0)


if __name__ == "__main__":
    unittest.main()
