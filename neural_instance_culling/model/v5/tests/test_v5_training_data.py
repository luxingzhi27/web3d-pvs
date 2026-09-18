from __future__ import annotations

import unittest

import numpy as np

from neural_instance_culling.model.v5.training_data import (
    _camera_basis,
    _query_geometry_numpy,
    _support_points,
)


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
