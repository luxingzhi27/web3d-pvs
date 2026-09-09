from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

BENCHMARK = Path(__file__).resolve().parents[1]
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))

from score_sidecar import (  # noqa: E402
    SIDECAR_SCHEMA,
    ScoreSidecarWriter,
    average_precision,
    read_score_sidecar,
)


class ScoreSidecarTests(unittest.TestCase):
    def test_typed_offsets_align_scores_and_ids_including_zero_gt_pose(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "scores"
            writer = ScoreSidecarWriter(
                root,
                split="test",
                threshold=0.5,
                checkpoint="/data/checkpoint.pt",
                calibration="/data/calibration_ready_summary.json",
            )
            writer.append_pose(
                10,
                np.asarray([7, 8, 9], dtype=np.uint32),
                np.asarray([0.5, 0.5, 0.1], dtype=np.float32),
                np.asarray([1, 0, 0], dtype=np.uint8),
                np.asarray([2.0, 0.0, 0.0], dtype=np.float32),
                np.asarray([7, 8], dtype=np.uint32),
            )
            writer.append_pose(
                11,
                np.asarray([12], dtype=np.uint32),
                np.asarray([0.2], dtype=np.float32),
                np.asarray([0], dtype=np.uint8),
                np.asarray([0.0], dtype=np.float32),
                np.zeros((0,), dtype=np.uint32),
            )
            writer.append_pose(
                12,
                np.zeros((0,), dtype=np.uint32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.uint8),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.uint32),
            )
            manifest_path = writer.close()

            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], SIDECAR_SCHEMA)
            self.assertEqual(payload["poseCount"], 3)
            self.assertEqual(payload["candidateCount"], 4)
            self.assertEqual(payload["predictedCount"], 2)
            self.assertEqual(payload["dtypes"]["poseOffsets"], "<u8")
            self.assertEqual(payload["dtypes"]["candidateIds"], "<u4")
            self.assertEqual(payload["dtypes"]["scores"], "<f4")

            arrays = read_score_sidecar(manifest_path)
            np.testing.assert_array_equal(arrays["poseIndices"], [10, 11, 12])
            np.testing.assert_array_equal(arrays["poseOffsets"], [0, 3, 4, 4])
            np.testing.assert_array_equal(arrays["predictedOffsets"], [0, 2, 2, 2])
            np.testing.assert_array_equal(arrays["candidateIds"], [7, 8, 9, 12])
            np.testing.assert_allclose(arrays["scores"], [0.5, 0.5, 0.1, 0.2])
            np.testing.assert_array_equal(arrays["predictedIds"], [7, 8])

    def test_average_precision_groups_equal_scores_and_returns_none_for_zero_gt(self) -> None:
        self.assertAlmostEqual(
            average_precision(
                np.asarray([0.5, 0.5, 0.1], dtype=np.float32),
                np.asarray([1, 0, 0], dtype=np.uint8),
            ),
            0.5,
        )
        self.assertIsNone(
            average_precision(
                np.asarray([0.8, 0.8], dtype=np.float32),
                np.asarray([0, 0], dtype=np.uint8),
            )
        )

    def test_reader_rejects_non_monotone_pose_indices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "scores"
            writer = ScoreSidecarWriter(
                root,
                split="validation",
                threshold=0.5,
            )
            writer.append_pose(
                2,
                np.asarray([1], dtype=np.uint32),
                np.asarray([0.5], dtype=np.float32),
                np.asarray([1], dtype=np.uint8),
                np.asarray([1.0], dtype=np.float32),
                np.asarray([1], dtype=np.uint32),
            )
            writer.append_pose(
                3,
                np.asarray([2], dtype=np.uint32),
                np.asarray([0.4], dtype=np.float32),
                np.asarray([0], dtype=np.uint8),
                np.asarray([0.0], dtype=np.float32),
                np.zeros((0,), dtype=np.uint32),
            )
            manifest = writer.close()
            pose_indices = np.fromfile(root / "poseIndices.bin", dtype="<i8")
            pose_indices[:] = [3, 2]
            pose_indices.tofile(root / "poseIndices.bin")
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                read_score_sidecar(manifest)


if __name__ == "__main__":
    unittest.main()
