from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from neural_instance_culling.model.common.candidate_identity import (
    audit_native_aabb_candidates,
)
from neural_instance_culling.model.pose_csr_dataset import (
    DIRECTIONAL_POSE_DTYPE,
    frustum_candidate_ids_for_pose,
)


class _Dataset:
    def __init__(self, root: Path, poses: np.ndarray, stored: np.ndarray) -> None:
        self.dataset_dir = root
        self.poses = poses
        self.meta = {
            "candidateSemantics": (
                "Union of full AABB candidates computed independently for every "
                "successful subpose; no GT positive union"
            ),
            "files": {
                "subposeOffsets": "subpose_offsets.bin",
                "subposeCameraPos": "subpose_camera_pos.bin",
            },
        }
        self._stored = stored

    def frustum_slice(self, pose_index: int) -> np.ndarray:
        if pose_index != 0:
            raise IndexError(pose_index)
        return self._stored


class CandidateIdentityTest(unittest.TestCase):
    def test_subpose_union_candidate_semantics_are_recomputed_as_a_union(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            poses = np.zeros((1,), dtype=DIRECTIONAL_POSE_DTYPE)
            poses["camera_forward"][0] = [0.0, 0.0, -1.0]
            poses["camera_view"][0] = [1.0, 1.0]
            positions = np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype="<f4")
            np.asarray([0, 2], dtype="<u8").tofile(root / "subpose_offsets.bin")
            positions.tofile(root / "subpose_camera_pos.bin")
            aabbs = np.asarray(
                [
                    [-2.0, -0.5, -4.0, -1.5, 0.5, -3.0],
                    [1.5, -0.5, -4.0, 2.0, 0.5, -3.0],
                    [20.0, -0.5, -4.0, 21.0, 0.5, -3.0],
                ],
                dtype=np.float32,
            )
            parts = [
                frustum_candidate_ids_for_pose(
                    position,
                    poses["camera_forward"][0],
                    1.0,
                    1.0,
                    aabbs,
                    near=0.05,
                )
                for position in positions
            ]
            stored = np.unique(np.concatenate(parts)).astype(np.uint32)
            summary = audit_native_aabb_candidates(
                _Dataset(root, poses, stored), aabbs, [0]
            )
            self.assertEqual(
                summary["source"],
                "recomputed_union_of_native_subpose_aabb_candidates",
            )
            self.assertFalse(summary["gtUnionUsed"])


if __name__ == "__main__":
    unittest.main()
