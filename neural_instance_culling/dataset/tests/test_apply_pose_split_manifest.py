from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from neural_instance_culling.dataset.apply_pose_split_manifest import (
    POSE_DTYPE,
    _materialize_viewcell_query_geometry,
)


class ApplyPoseSplitManifestTest(unittest.TestCase):
    def test_viewcell_geometry_keeps_query_and_backed_candidate_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            output.mkdir()
            centers = np.asarray([[1.0, 2.0, 3.0], [5.0, 2.0, 7.0]], dtype="<f4")
            forwards = np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype="<f4")
            params = np.zeros((2, 8), dtype="<f4")
            params[:, 0] = 2.5
            centers.tofile(source / "viewcell_centers.bin")
            forwards.tofile(source / "viewcell_forwards.bin")
            params.tofile(source / "viewcell_params.bin")
            poses = np.zeros((2,), dtype=POSE_DTYPE)
            poses["camera_forward"] = forwards / np.linalg.norm(
                forwards, axis=1, keepdims=True
            )

            metadata = _materialize_viewcell_query_geometry(source, output, poses)
            query = np.fromfile(output / "query_center_world.bin", dtype="<f4").reshape(-1, 3)
            candidate = np.fromfile(
                output / "candidate_camera_world.bin", dtype="<f4"
            ).reshape(-1, 3)
            expected_offset = 2.5 / math.tan(math.radians(30.0))
            np.testing.assert_allclose(query, centers)
            np.testing.assert_allclose(
                np.linalg.norm(query - candidate, axis=1),
                expected_offset,
                rtol=1e-6,
                atol=1e-6,
            )
            np.testing.assert_allclose(poses["camera_world"], candidate)
            self.assertEqual(
                metadata["candidateBackOffsetDefinition"],
                "viewcell_radius / tan(frontend_render_fov_y_deg / 2), frontend_render_fov_y_deg=60",
            )


if __name__ == "__main__":
    unittest.main()
