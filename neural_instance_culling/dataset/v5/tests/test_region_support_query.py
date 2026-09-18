from __future__ import annotations

import unittest

import numpy as np

from neural_instance_culling.dataset.v5.region_support_query import (
    build_oriented_box_support_points,
    build_region_support_manifest,
    build_region_support_points,
    query_region_stats,
)


class RegionSupportQueryTest(unittest.TestCase):
    def test_disk_has_center_and_eight_equal_angle_points(self) -> None:
        manifest = build_region_support_manifest(
            region_type="disk", center=[1.0, 2.0, 3.0], right=[1.0, 0.0, 0.0], forward=[0.0, 0.0, 1.0], radius=2.0
        )
        points = build_region_support_points(manifest)
        self.assertEqual(points.shape, (9, 3))
        self.assertTrue(np.allclose(points[0], [1.0, 2.0, 3.0]))
        self.assertTrue(np.allclose(np.linalg.norm(points[1:] - points[0], axis=1), 2.0))

    def test_oriented_box_has_center_and_eight_corners(self) -> None:
        points = build_oriented_box_support_points([0, 0, 0], [[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        self.assertEqual(points.shape, (9, 3))
        self.assertEqual({tuple(row) for row in points[1:]}, {
            (-1.0, -2.0, -3.0), (-1.0, -2.0, 3.0), (-1.0, 2.0, -3.0), (-1.0, 2.0, 3.0),
            (1.0, -2.0, -3.0), (1.0, -2.0, 3.0), (1.0, 2.0, -3.0), (1.0, 2.0, 3.0),
        })

    def test_stats_order_is_fixed(self) -> None:
        stats = query_region_stats(np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32))
        self.assertTrue(np.allclose(stats, [1.0, 9.0, 5.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
