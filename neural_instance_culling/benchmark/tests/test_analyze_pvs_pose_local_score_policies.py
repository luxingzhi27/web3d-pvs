from __future__ import annotations

import unittest

from neural_instance_culling.benchmark.analyze_pvs_pose_local_score_policies import (
    _transform_pose_scores,
    analyze,
)

import numpy as np


def _capture(split: str) -> dict:
    return {
        "split": split,
        "testRead": False,
        "perPose": [
            {
                "candidateScores": [0.9, 0.8, 0.2, 0.1],
                "targets": [1, 1, 0, 0],
                "visibleWeights": [10.0, 1.0, 0.0, 0.0],
            },
            {
                "candidateScores": [0.09, 0.08, 0.02, 0.01],
                "targets": [1, 1, 0, 0],
                "visibleWeights": [10.0, 1.0, 0.0, 0.0],
            },
        ],
    }


class PoseLocalScorePolicyTest(unittest.TestCase):
    def test_robust_transform_removes_pose_logit_location_and_scale(self) -> None:
        first = _transform_pose_scores(np.asarray([0.9, 0.8, 0.2, 0.1]), "robust_logit")
        second = _transform_pose_scores(np.asarray([0.09, 0.08, 0.02, 0.01]), "robust_logit")
        self.assertAlmostEqual(float(np.median(first)), 0.0)
        self.assertAlmostEqual(float(np.median(second)), 0.0)
        self.assertGreater(float(first[0]), float(first[-1]))
        self.assertGreater(float(second[0]), float(second[-1]))

    def test_analysis_freezes_each_deployable_policy_on_calibration(self) -> None:
        payload = analyze(
            _capture("calibration"),
            _capture("validation"),
            bootstrap_replicates=100,
            seed=7,
        )
        self.assertFalse(payload["testRead"])
        self.assertEqual(payload["calibrationPoseCount"], 2)
        self.assertEqual(payload["validationPoseCount"], 2)
        for policy in payload["policies"].values():
            self.assertTrue(policy["deployable"])
            self.assertGreater(policy["calibration"]["weightedRecall"], 0.99)
            self.assertGreater(
                policy["calibration"]["weightedRecallLowerConfidenceBound"],
                0.99,
            )
        self.assertFalse(
            payload["oracleUpperBounds"]["poseWeightedRecall99"].get(
                "deployable", False
            )
        )


if __name__ == "__main__":
    unittest.main()
