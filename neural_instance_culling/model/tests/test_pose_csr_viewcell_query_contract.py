from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from neural_instance_culling.model.pose_csr_dataset import (
    DIRECTIONAL_POSE_DTYPE,
    PoseCSRDataset,
    PoseCSRSplit,
)


class PoseCSRViewCellQueryContractTest(unittest.TestCase):
    def test_capped_pose_set_batches_do_not_repeat_before_exhaustion(self) -> None:
        class DatasetStub:
            visible_counts = np.ones((20,), dtype=np.int64)
            frustum_counts = np.ones((20,), dtype=np.int64)

        split = PoseCSRSplit(
            DatasetStub(),
            "diagnostic",
            pose_indices=np.arange(20, dtype=np.int64),
        )
        batches = list(
            split.pose_set_batches(
                4,
                np.random.default_rng(20260817),
                max_steps=3,
                include_empty=False,
            )
        )
        selected = np.concatenate(batches)
        self.assertEqual(selected.size, 12)
        self.assertEqual(np.unique(selected).size, 12)

    def test_capped_pose_set_batches_repeat_only_after_a_full_cycle(self) -> None:
        class DatasetStub:
            visible_counts = np.ones((5,), dtype=np.int64)
            frustum_counts = np.ones((5,), dtype=np.int64)

        split = PoseCSRSplit(
            DatasetStub(),
            "diagnostic",
            pose_indices=np.arange(5, dtype=np.int64),
        )
        selected = np.concatenate(
            list(
                split.pose_set_batches(
                    2,
                    np.random.default_rng(20260817),
                    max_steps=4,
                    include_empty=False,
                )
            )
        )
        self.assertEqual(selected.size, 7)
        self.assertEqual(np.unique(selected[:5]).size, 5)

    def test_capped_pose_set_batches_keep_a_short_last_batch(self) -> None:
        class DatasetStub:
            visible_counts = np.ones((5,), dtype=np.int64)
            frustum_counts = np.ones((5,), dtype=np.int64)

        split = PoseCSRSplit(
            DatasetStub(),
            "diagnostic",
            pose_indices=np.arange(5, dtype=np.int64),
        )
        batches = list(
            split.pose_set_batches(
                2,
                np.random.default_rng(20260817),
                max_steps=3,
                include_empty=False,
            )
        )
        self.assertEqual([batch.size for batch in batches], [2, 2, 1])
        self.assertEqual(np.unique(np.concatenate(batches)).size, 5)

    def test_hard_pose_batches_mix_registered_tail_and_uniform_poses(self) -> None:
        class DatasetStub:
            visible_counts = np.ones((20,), dtype=np.int64)
            frustum_counts = np.ones((20,), dtype=np.int64)

        split = PoseCSRSplit(
            DatasetStub(),
            "diagnostic",
            pose_indices=np.arange(20, dtype=np.int64),
        )
        hard = np.asarray([0, 1, 2, 3], dtype=np.int64)
        batches = list(
            split.pose_set_batches(
                4,
                np.random.default_rng(20260818),
                max_steps=8,
                hard_pose_indices=hard,
                hard_pose_fraction=0.5,
            )
        )

        self.assertEqual(len(batches), 8)
        for batch in batches:
            self.assertEqual(batch.size, 4)
            self.assertEqual(np.unique(batch).size, 4)
            self.assertEqual(np.isin(batch, hard).sum(), 2)

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

    def test_subpose_sidecar_accepts_a_split_only_dataset_view(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "split_view"
            sidecar = Path(directory) / "sidecar"
            root.mkdir()
            sidecar.mkdir()
            (root / "dataset_meta.json").write_text(
                json.dumps(
                    {
                        "poseCount": 1,
                        "poseStrideBytes": int(DIRECTIONAL_POSE_DTYPE.itemsize),
                        "splitIds": {"train": 0},
                        "visibleWeightDtype": "float32",
                        "sourceDataset": str(Path(directory) / "original_view"),
                    }
                ),
                encoding="utf-8",
            )
            poses = np.zeros((1,), dtype=DIRECTIONAL_POSE_DTYPE)
            poses["split"][0] = 0
            poses.tofile(root / "poses.bin")
            np.asarray([0, 1], dtype="<u8").tofile(root / "visible_offsets.bin")
            np.asarray([0], dtype="<u4").tofile(root / "visible_ids.bin")
            np.asarray([1.0], dtype="<f4").tofile(root / "visible_weights.bin")
            np.asarray([0, 1], dtype="<u8").tofile(root / "frustum_offsets.bin")
            np.asarray([0], dtype="<u4").tofile(root / "frustum_ids.bin")
            (sidecar / "sidecar_manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "pvs-viewcell-subpose-supervision-sidecar-v1",
                        "poseCount": 1,
                        "mainCsrDataset": str(Path(directory) / "original_view"),
                        "splitLabelsMayDiffer": True,
                    }
                ),
                encoding="utf-8",
            )
            np.asarray([1], dtype="<u2").tofile(sidecar / "visible_hit_counts.bin")
            np.asarray([0, 1], dtype="<u8").tofile(sidecar / "subpose_offsets.bin")

            dataset = PoseCSRDataset(root, num_instances=1, subpose_sidecar=sidecar)

            self.assertTrue(dataset.has_subpose_robust_labels)
            self.assertTrue(dataset.subpose_sidecar_meta["splitLabelsMayDiffer"])


if __name__ == "__main__":
    unittest.main()
