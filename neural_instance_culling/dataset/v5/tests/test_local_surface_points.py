from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from neural_instance_culling.dataset.v5.build_local_surface_points import (
    POINTS_PER_UNIT,
    SURFACE_BINARY_HEADER,
    SurfaceUnitInput,
    build_local_surface_asset,
    load_local_surface_asset,
    write_local_surface_asset,
)


def _triangle_unit(unit_id: int, *, degenerate: bool = False) -> SurfaceUnitInput:
    if degenerate:
        vertices = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    else:
        vertices = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    transform = np.asarray(
        [[2.0, 0.0, 0.0, 3.0], [0.0, 3.0, 0.0, -2.0], [0.0, 0.0, 4.0, 5.0], [0.0, 0.0, 0.0, 1.0]]
    )
    return SurfaceUnitInput(unit_id, vertices, np.asarray([[0, 1, 2]]), transform)


class LocalSurfacePointsTest(unittest.TestCase):
    def test_dense_binary_contract_and_inverse_transpose_normals(self) -> None:
        asset = build_local_surface_asset([_triangle_unit(4), _triangle_unit(1)], sampling_seed=17)
        self.assertEqual(asset.points.shape, (2, POINTS_PER_UNIT, 6))
        self.assertEqual(asset.points.dtype, np.float32)
        self.assertTrue(np.allclose(np.linalg.norm(asset.points[:, :, 3:], axis=2), 1.0, atol=1e-5))
        self.assertTrue(np.allclose(asset.points[:, :, 3:], [0.0, 0.0, 1.0], atol=1e-5))
        self.assertTrue(np.allclose(asset.size_ratios[0], [2.0 / 3.0, 1.0, 0.0], atol=1e-6))
        with tempfile.TemporaryDirectory() as temp:
            manifest_path = write_local_surface_asset(asset, temp)
            loaded = load_local_surface_asset(Path(temp))
            self.assertEqual(manifest_path.name, "surface_manifest.json")
            self.assertEqual((Path(temp) / "surface_points_fp32.bin").read_bytes()[:4], b"GPV5")
            self.assertEqual(SURFACE_BINARY_HEADER.size, 16)
            self.assertEqual(
                (Path(temp) / "surface_points_fp32.bin").stat().st_size,
                16 + 2 * POINTS_PER_UNIT * 6 * 4,
            )
            self.assertEqual((Path(temp) / "size_ratios_fp32.bin").stat().st_size, 16 + 2 * 3 * 4)
            self.assertTrue(np.array_equal(loaded.points, asset.points))
            self.assertTrue(np.array_equal(loaded.size_ratios, asset.size_ratios))

    def test_degenerate_unit_keeps_dense_zero_row_and_manifest_marker(self) -> None:
        asset = build_local_surface_asset([_triangle_unit(0), _triangle_unit(1, degenerate=True)], sampling_seed=1)
        self.assertEqual(asset.points.shape, (2, POINTS_PER_UNIT, 6))
        self.assertTrue(np.all(asset.points[1] == 0.0))
        self.assertEqual(asset.degenerate_unit_ids, (1,))
        with tempfile.TemporaryDirectory() as temp:
            write_local_surface_asset(asset, temp)
            loaded = load_local_surface_asset(temp)
            self.assertEqual(loaded.degenerate_unit_ids, (1,))
            self.assertTrue(np.all(loaded.points[1] == 0.0))


if __name__ == "__main__":
    unittest.main()
