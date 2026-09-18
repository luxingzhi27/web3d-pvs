from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from neural_instance_culling.benchmark.v5.calibration import (
    calibrate_source_global,
    calibrate_target,
    select_calibration_workpoint,
)
from neural_instance_culling.benchmark.v5.contracts import PoseRecord
from neural_instance_culling.benchmark.v5.metrics import evaluate_scene


def _pose(
    pose_id: str,
    scores: list[float],
    targets: list[int],
    weights: list[float],
    *,
    split: str = "validation",
    scene: str = "fixture",
) -> PoseRecord:
    return PoseRecord(
        pose_id=pose_id,
        split=split,
        candidate_ids=list(range(len(scores))),
        scores=scores,
        targets=targets,
        visible_weights=weights,
        scene=scene,
    )


class V5MetricsTests(unittest.TestCase):
    def test_reported_denominators_and_pose_ap_are_distinct(self) -> None:
        rows = [
            _pose("p0", [0.9, 0.2, 0.8], [1, 0, 1], [2.0, 0.0, 1.0]),
            _pose("p1", [0.7, 0.1], [0, 0], [0.0, 0.0]),
        ]
        result = evaluate_scene(rows, 0.85, bootstrap_replicates=64, bootstrap_seed=7)

        self.assertAlmostEqual(result["weighted_recall"], 2.0 / 3.0)
        self.assertAlmostEqual(result["ordinary_recall"], 0.5)
        self.assertAlmostEqual(result["fn_over_gt"], 0.5)
        self.assertAlmostEqual(result["bad_cull"], 1.0 / 5.0)
        self.assertAlmostEqual(result["useful_cull"], 3.0 / 5.0)
        self.assertAlmostEqual(result["fp_over_gt"], 0.0)
        self.assertAlmostEqual(result["pred_over_gt"], 0.5)
        self.assertAlmostEqual(result["cnor"], 1.0)
        self.assertAlmostEqual(result["pose_pr_auc"], 1.0)
        self.assertAlmostEqual(result["pose_prevalence"], 2.0 / 3.0)
        self.assertAlmostEqual(result["pose_ap_lift"], 1.5)
        self.assertIsNotNone(result["weighted_recall_lcb"])

    def test_calibration_strict_tier_precedes_higher_mean_only_tier(self) -> None:
        rows = [
            {
                "threshold": 0.8,
                "selection_split": "calibration",
                "test_read": False,
                "scene_metrics": {
                    "scene": {"weighted_recall": 0.999, "weighted_recall_lcb": 0.989}
                },
            },
            {
                "threshold": 0.7,
                "selection_split": "calibration",
                "test_read": False,
                "scene_metrics": {
                    "scene": {"weighted_recall": 0.995, "weighted_recall_lcb": 0.991}
                },
            },
        ]
        selected = select_calibration_workpoint(rows, protocol="target_calibrated")
        self.assertEqual(selected.status, "strict_lcb_target")
        self.assertEqual(selected.threshold, 0.7)

    def test_calibration_falls_back_to_mean_target_without_calling_it_strict(self) -> None:
        rows = [
            {
                "threshold": 0.8,
                "selection_split": "calibration",
                "test_read": False,
                "scene_metrics": {
                    "scene": {"weighted_recall": 0.995, "weighted_recall_lcb": 0.989}
                },
            },
            {
                "threshold": 0.7,
                "selection_split": "calibration",
                "test_read": False,
                "scene_metrics": {
                    "scene": {"weighted_recall": 0.98, "weighted_recall_lcb": 0.97}
                },
            },
        ]
        selected = select_calibration_workpoint(rows, protocol="target_calibrated")
        self.assertEqual(selected.status, "mean_target")
        self.assertTrue(selected.mean_target_met)
        self.assertFalse(selected.confidence_target_met)
        self.assertEqual(selected.threshold, 0.8)

    def test_test_rows_cannot_enter_threshold_selection(self) -> None:
        test_row = _pose("test", [0.9], [1], [1.0], split="test")
        with self.assertRaisesRegex(ValueError, "calibration"):
            calibrate_target([test_row])

    def test_source_global_uses_only_explicit_source_calibration(self) -> None:
        source = {
            "source_a": [_pose("a", [0.9], [1], [1.0], split="calibration", scene="source_a")],
            "source_b": [_pose("b", [0.9], [1], [1.0], split="calibration", scene="source_b")],
        }
        selection = calibrate_source_global(source, bootstrap_replicates=16)
        self.assertEqual(selection.protocol, "source_global")
        self.assertEqual(selection.selection_split, "calibration")
        self.assertFalse(selection.test_read)
        self.assertEqual(set(selection.scene_metrics), {"source_a", "source_b"})

    def test_exact_calibration_bootstrap_is_logarithmic_in_change_points(self) -> None:
        rows = []
        for pose_index in range(32):
            scores = np.linspace(
                pose_index * 1e-5,
                1.0 + pose_index * 1e-5,
                128,
                dtype=np.float64,
            )
            targets = np.zeros((128,), dtype=np.uint8)
            targets[::2] = 1
            rows.append(
                _pose(
                    f"p{pose_index}",
                    scores.tolist(),
                    targets.tolist(),
                    targets.astype(np.float64).tolist(),
                    split="calibration",
                )
            )
        with mock.patch(
            "neural_instance_culling.benchmark.v5.calibration._weighted_lcb",
            wraps=__import__(
                "neural_instance_culling.benchmark.v5.calibration",
                fromlist=["_weighted_lcb"],
            )._weighted_lcb,
        ) as lcb:
            selection = calibrate_target(rows, scene="fixture", bootstrap_replicates=32)
        self.assertEqual(selection.status, "strict_lcb_target")
        self.assertLessEqual(lcb.call_count, 16)


if __name__ == "__main__":
    unittest.main()
