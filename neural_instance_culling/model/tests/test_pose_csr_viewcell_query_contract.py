from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from neural_instance_culling.model.pose_csr_dataset import (
    DIRECTIONAL_POSE_DTYPE,
    PoseCSRDataset,
)


class PoseCSRViewCellQueryContractTest(unittest.TestCase):
    def test_batch_keeps_candidate_camera_and_query_center_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta = {
                "schema": "pose-csr-viewcell-back-camera-color-id-fov66-v2",
                "poseCount": 1,
                "poseStrideBytes": int(DIRECTIONAL_POSE_DTYPE.itemsize),
                "splitIds": {"train": 0},
                "visibleWeightDtype": "float32",
                "files": {
                    "queryCenterWorld": "query_center_world.bin",
                    "viewcellRadiusM": "viewcell_radius_m.bin",
                },
            }
            (root / "dataset_meta.json").write_text(json.dumps(meta), encoding="utf-8")
            poses = np.zeros((1,), dtype=DIRECTIONAL_POSE_DTYPE)
            poses["camera_world"][0] = [0.0, 0.0, 3.4641016]
            poses["camera_forward"][0] = [0.0, 0.0, -1.0]
            poses["camera_view"][0] = [1.1, 0.65]
            poses["split"][0] = 0
            poses.tofile(root / "poses.bin")
            np.asarray([0, 1], dtype="<u8").tofile(root / "visible_offsets.bin")
            np.asarray([0], dtype="<u4").tofile(root / "visible_ids.bin")
            np.asarray([1.0], dtype="<f4").tofile(root / "visible_weights.bin")
            np.asarray([0, 1], dtype="<u8").tofile(root / "frustum_offsets.bin")
            np.asarray([0], dtype="<u4").tofile(root / "frustum_ids.bin")
            np.asarray([[0.0, 0.0, 0.0]], dtype="<f4").tofile(root / "query_center_world.bin")
            np.asarray([2.0], dtype="<f4").tofile(root / "viewcell_radius_m.bin")

            dataset = PoseCSRDataset(root, num_instances=1)
            batch = dataset.split("train").build_pose_set_batch(
                np.asarray([0]),
                np.asarray([[-1.0, -1.0, -2.0, 1.0, 1.0, 0.0]], dtype=np.float32),
                np.random.default_rng(1),
            )
            self.assertTrue(np.allclose(batch["candidate_camera_world"][0], [0.0, 0.0, 3.4641016]))
            self.assertTrue(np.allclose(batch["query_center_world"][0], [0.0, 0.0, 0.0]))
            self.assertEqual(float(batch["viewcell_radius_m"][0]), 2.0)

    def test_required_query_center_rejects_legacy_dataset(self) -> None:
        # The explicit required flag is the v3 training gate; old models may
        # still read historical PoseCSR files without silently entering v3.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dataset_meta.json").write_text(
                json.dumps({
                    "poseCount": 1,
                    "poseStrideBytes": int(DIRECTIONAL_POSE_DTYPE.itemsize),
                    "splitIds": {"train": 0},
                    "visibleWeightDtype": "float32",
                }),
                encoding="utf-8",
            )
            poses = np.zeros((1,), dtype=DIRECTIONAL_POSE_DTYPE)
            poses["split"][0] = 0
            poses.tofile(root / "poses.bin")
            np.asarray([0, 0], dtype="<u8").tofile(root / "visible_offsets.bin")
            np.asarray([], dtype="<u4").tofile(root / "visible_ids.bin")
            np.asarray([], dtype="<f4").tofile(root / "visible_weights.bin")
            np.asarray([0, 0], dtype="<u8").tofile(root / "frustum_offsets.bin")
            np.asarray([], dtype="<u4").tofile(root / "frustum_ids.bin")
            dataset = PoseCSRDataset(root, num_instances=1)
            with self.assertRaisesRegex(ValueError, "query_center_world"):
                dataset.query_center_world(0, required=True)


if __name__ == "__main__":
    unittest.main()
