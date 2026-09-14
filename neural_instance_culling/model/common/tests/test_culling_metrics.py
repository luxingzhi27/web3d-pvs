from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[2]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from common.culling_metrics import candidate_normalized_occlusion_recall


class CandidateNormalizedOcclusionRecallTest(unittest.TestCase):
    def test_weights_pose_occlusion_recall_by_normalized_negative_opportunity(self) -> None:
        value = candidate_normalized_occlusion_recall(
            np.asarray([10.0, 80.0]),
            np.asarray([0.0, 20.0]),
            np.asarray([100.0, 1000.0]),
        )
        self.assertAlmostEqual(value, 0.9)

    def test_ignores_poses_without_candidate_or_negative_opportunity(self) -> None:
        value = candidate_normalized_occlusion_recall(
            np.asarray([0.0, 0.0, 4.0]),
            np.asarray([0.0, 0.0, 1.0]),
            np.asarray([0.0, 7.0, 10.0]),
        )
        self.assertAlmostEqual(value, 0.8)

    def test_returns_one_when_no_pose_has_occlusion_opportunity(self) -> None:
        value = candidate_normalized_occlusion_recall(
            np.asarray([0.0, 0.0]),
            np.asarray([0.0, 0.0]),
            np.asarray([0.0, 12.0]),
        )
        self.assertEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
