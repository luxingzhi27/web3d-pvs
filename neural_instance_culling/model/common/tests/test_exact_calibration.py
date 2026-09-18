from __future__ import annotations

import unittest

import numpy as np

from neural_instance_culling.model.common.exact_calibration import (
    FixedPoseBootstrap,
    float32_score_change_points,
    select_highest_recall_target_score_change_point,
)


class ExactCalibrationTest(unittest.TestCase):
    def test_change_points_preserve_float32_values(self) -> None:
        values = float32_score_change_points(
            np.asarray([0.1, 0.100000001, 0.2], dtype=np.float64)
        )
        np.testing.assert_array_equal(
            values,
            np.unique(np.asarray([0.1, 0.100000001, 0.2], dtype=np.float32)),
        )
        self.assertEqual(values.dtype, np.dtype(np.float32))

    def test_fixed_pose_bootstrap_is_deterministic_and_chunked(self) -> None:
        gt_mass = np.asarray([2.0, 1.0, 0.0, 3.0])
        plan = FixedPoseBootstrap.from_gt_mass(
            gt_mass,
            replicates=17,
            seed=19,
            chunk_size=3,
        )
        weighted_tp = np.asarray([2.0, 0.5, 0.0, 2.0])
        first = plan.lower_confidence_bound(weighted_tp, gt_mass)
        second = plan.lower_confidence_bound(weighted_tp, gt_mass)
        self.assertEqual(first, second)
        self.assertEqual(plan.metadata()["indexStorage"], "deterministic_chunked_generation")
        self.assertEqual(plan.metadata()["validPoseCount"], 3)

    def test_search_domain_includes_negative_only_change_point(self) -> None:
        """The search domain must not stop at the highest positive score."""
        scores = np.asarray([0.6, 0.7, 0.8], dtype=np.float32)
        labels = np.asarray([1.0, 1.0, 0.0], dtype=np.float32)
        weights = np.asarray([1.0, 1.0, 0.0], dtype=np.float32)
        selection = select_highest_recall_target_score_change_point(
            scores,
            labels,
            weights,
            np.asarray([0, 3], dtype=np.int64),
            bootstrap_replicates=8,
            bootstrap_seed=23,
            # This regression isolates the score-domain search.  The formal
            # positive safety target is supplied by the production caller.
            target_weighted_recall=-0.01,
        )
        self.assertEqual(selection["status"], "confidence_target_met")
        self.assertEqual(selection["threshold"], float(np.float32(0.8)))
        self.assertEqual(selection["selectedCandidateIndex"], 2)
        self.assertEqual(selection["allScoreChangePointCount"], 3)
        self.assertEqual(selection["positiveScoreChangePointCount"], 2)

    def test_next_higher_is_immediate_all_score_point(self) -> None:
        scores = np.asarray([0.9, 0.8, 0.7, 0.1], dtype=np.float32)
        labels = np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        weights = np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        selection = select_highest_recall_target_score_change_point(
            scores,
            labels,
            weights,
            np.asarray([0, 4], dtype=np.int64),
            bootstrap_replicates=8,
            bootstrap_seed=29,
        )
        self.assertEqual(selection["threshold"], np.float32(0.7))
        self.assertEqual(selection["nextHigherThreshold"], np.float32(0.8))
        self.assertEqual(selection["nextHigherSafety"]["meanTargetMet"], False)


if __name__ == "__main__":
    unittest.main()
