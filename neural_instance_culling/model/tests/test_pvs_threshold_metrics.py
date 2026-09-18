from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np
import torch

MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from pvs_threshold_metrics import evaluate_exact_calibration, evaluate_thresholds  # noqa: E402


class _FakeSplit:
    def pose_set_batches(self, _poses_per_batch, _rng, _max_steps, include_empty=True):
        self.include_empty = include_empty
        yield np.asarray([0, 1], dtype=np.int64)

    def build_pose_set_batch(
        self,
        _pose_indices,
        _world_aabbs,
        _rng,
        max_candidates_per_pose,
        allow_candidate_visible_union,
        include_empty=True,
    ):
        self.build_arguments = (
            max_candidates_per_pose,
            allow_candidate_visible_union,
            include_empty,
        )
        return {
            "pose_offsets": np.asarray([0, 2, 5], dtype=np.int64),
            "visible_counts": np.asarray([1, 2], dtype=np.int64),
            "instance": np.asarray([0, 1, 2, 3, 4], dtype=np.uint32),
            "target": np.asarray([1, 0, 1, 1, 0], dtype=np.uint8),
            "visible_weights": np.asarray([1, 0, 2, 1, 0], dtype=np.float32),
            "camera": np.zeros((5, 12), dtype=np.float32),
            "camera_world": np.zeros((5, 3), dtype=np.float32),
            "camera_view": np.zeros((5, 3), dtype=np.float32),
            "query_center_world": np.zeros((5, 3), dtype=np.float32),
            "viewcell_radius_m": np.ones((5, 1), dtype=np.float32),
        }


class _FakeModel:
    def eval(self):
        return self

    def compute_visibility_logits(
        self,
        _camera,
        _view,
        _camera_world,
        instance_ids,
        *,
        runtime_features,
        query_center_world,
        viewcell_radius_m,
        pose_offsets,
    ):
        del runtime_features, query_center_world, viewcell_radius_m, pose_offsets
        scores = torch.tensor([0.9, 0.4, 0.8, 0.7, 0.1], device=instance_ids.device)
        return torch.logit(scores[instance_ids.long()])


class ExactThresholdMetricsTest(unittest.TestCase):
    def test_exact_calibration_rejects_test_split(self) -> None:
        split = _FakeSplit()
        split.split_name = "test"
        with self.assertRaisesRegex(ValueError, "test split"):
            evaluate_exact_calibration(
                _FakeModel(),
                split,
                torch.zeros((5, 1), dtype=torch.float32),
                np.zeros((5, 6), dtype=np.float32),
                torch.device("cpu"),
                poses_per_batch=2,
                max_steps=None,
                max_candidates_per_pose=0,
                seed=7,
                bootstrap_replicates=8,
            )

    def test_exact_calibration_runs_one_score_pass_and_returns_change_point(self) -> None:
        split = _FakeSplit()
        rows = evaluate_exact_calibration(
            _FakeModel(),
            split,
            torch.zeros((5, 1), dtype=torch.float32),
            np.zeros((5, 6), dtype=np.float32),
            torch.device("cpu"),
            poses_per_batch=2,
            max_steps=None,
            max_candidates_per_pose=0,
            seed=7,
            bootstrap_replicates=8,
            collect_score_stats=True,
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertAlmostEqual(row["threshold"], float(np.float32(0.7)), places=7)
        self.assertTrue(row["thresholdIsScoreChangePoint"])
        self.assertEqual(row["thresholdSource"], "calibration_actual_float32_score_change_points")
        self.assertEqual(row["thresholdSelection"]["nextHigherThreshold"], float(np.float32(0.8)))
        self.assertEqual(row["aggregateWeightedRecall"], 1.0)
        self.assertEqual(row["aggregateWeightedRecallLowerConfidenceBound"], 1.0)
        self.assertEqual(row["scoreDistribution"]["scoreCount"], 5)
        self.assertEqual(split.build_arguments, (0, False, True))

    def test_none_threshold_is_the_explicit_exact_calibration_mode(self) -> None:
        rows = evaluate_thresholds(
            _FakeModel(),
            _FakeSplit(),
            torch.zeros((5, 1), dtype=torch.float32),
            np.zeros((5, 6), dtype=np.float32),
            torch.device("cpu"),
            poses_per_batch=2,
            max_steps=None,
            max_candidates_per_pose=0,
            seed=7,
            thresholds=None,
            collect_pose_stats=True,
            bootstrap_replicates=8,
            collect_score_stats=True,
            target_weighted_recall=0.997,
        )
        self.assertEqual(rows[0]["thresholdSelection"]["thresholdDtype"], "float32")
        self.assertEqual(rows[0]["thresholdSelection"]["targetWeightedRecall"], 0.997)


if __name__ == "__main__":
    unittest.main()
