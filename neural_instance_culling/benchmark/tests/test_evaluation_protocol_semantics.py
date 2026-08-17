from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BENCHMARK_DIR.parent / "model"
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(MODEL_DIR))

import evaluate_proxy_interventions as interventions  # noqa: E402
import evaluate_visual_utility_metrics as utility_metrics  # noqa: E402


class _FixtureSplit:
    pose_indices = np.asarray([0], dtype=np.int64)

    def __init__(self, visible_weights: np.ndarray):
        self.visible_weights = np.asarray(visible_weights, dtype=np.float32)

    def pose_set_batches(self, poses_per_batch, rng, max_steps, include_empty=True):
        del poses_per_batch, rng, max_steps, include_empty
        yield np.asarray([0], dtype=np.int64)

    def build_pose_set_batch(self, pose_indices, world_aabbs, rng, max_candidates_per_pose=0, include_empty=True):
        del pose_indices, world_aabbs, rng, max_candidates_per_pose, include_empty
        return {
            "camera": np.zeros((2, 3), dtype=np.float32),
            "camera_world": np.zeros((2, 3), dtype=np.float32),
            "camera_view": np.ones((2, 5), dtype=np.float32),
            "instance": np.asarray([0, 1], dtype=np.int64),
            "target": np.asarray([1.0, 0.0], dtype=np.float32),
            "visible_weights": self.visible_weights,
            "pose_offsets": np.asarray([0, 2], dtype=np.int64),
            "visible_counts": np.asarray([1], dtype=np.int64),
            "candidate_counts": np.asarray([2], dtype=np.int64),
        }


class _FixtureRunner:
    name = "fixture"
    kind = "fixture"
    information_level = "fixture"
    decision_mode = "fixture"
    world_aabbs = np.zeros((2, 6), dtype=np.float32)
    instance_to_glb = np.asarray([0, 1], dtype=np.int64)

    def score_batch(self, batch):
        del batch
        return SimpleNamespace(
            scores=np.asarray([0.8, 0.2], dtype=np.float32),
            utility_scores=np.asarray([0.7, 0.1], dtype=np.float32),
            download_scores=np.asarray([0.9, 0.1], dtype=np.float32),
            forward_ms=0.1,
            total_ms=0.2,
            diagnostics={},
        )


def _evaluate_fixture(weights: np.ndarray) -> dict:
    return utility_metrics.evaluate_runner(
        _FixtureRunner(),
        _FixtureSplit(weights),
        np.asarray([0.5], dtype=np.float32),
        (1,),
        (10.0,),
        None,
        (),
        np.asarray([10.0, 20.0], dtype=np.float64),
        poses_per_batch=1,
        max_eval_poses=0,
        max_candidates_per_pose=0,
        seed=7,
        target_utility_recall=0.98,
        target_weighted_recall=0.99,
        score_modes=("visibility-only",),
        aggregations=("max",),
        aggregation_top_k=1,
        sample_with_replacement=False,
        progress=False,
    )


class EvaluationProtocolSemanticsTests(unittest.TestCase):
    def test_budget_aggregation_ignores_non_applicable_none_values(self) -> None:
        averaged = utility_metrics._average_rows_with_status(
            [
                {"utilityRecall": 0.8, "requiredStatus": "available"},
                {"utilityRecall": None, "requiredStatus": "not_applicable"},
            ]
        )
        self.assertEqual(averaged["utilityRecall"], 0.8)
        self.assertEqual(averaged["requiredStatus"], "mixed")

    def test_dynamic_learned_model_spec_is_explicit_and_structured(self) -> None:
        specs = utility_metrics.parse_learned_model_specs(
            ["formal|/tmp/model.pt|/tmp/features.bin|/tmp/calibration_ready_summary.json"]
        )
        self.assertEqual(specs["formal"]["kind"], "directional_occlusion_proxy_encoder")
        v3 = utility_metrics.parse_learned_model_specs(
            ["formal-v3|bounded_relation_survival_moment_v4|/tmp/model.pt|/tmp/features.bin|/tmp/calibration.json"]
        )
        self.assertEqual(v3["formal-v3"]["kind"], "bounded_relation_survival_moment_v4")
        with self.assertRaisesRegex(ValueError, "name\|checkpoint"):
            utility_metrics.parse_learned_model_specs(["formal|only-two-fields"])

    def test_invalid_visible_weights_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _evaluate_fixture(np.asarray([-1.0, 0.0], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            _evaluate_fixture(np.asarray([np.nan, 0.0], dtype=np.float32))

    def test_zero_utility_pose_is_not_applicable(self) -> None:
        summary = _evaluate_fixture(np.asarray([0.0, 0.0], dtype=np.float32))
        row = summary["thresholdRows"][0]
        self.assertIsNone(row["pose_visual_utility_recall"])
        self.assertIsNone(row["pose_miss_visual_utility_rate"])
        self.assertEqual(summary["utilityValidPoseCount"], 0)
        self.assertEqual(summary["utilityNotApplicablePoseCount"], 1)
        budget = summary["scoreModes"]["visibility-only"]["aggregations"]["max"]
        self.assertEqual(budget["bytePrefixAtUtilityTarget"]["utilityStatus"], "not_applicable")

    def test_m3_prefers_pre_test_calibration_record_and_rejects_test_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "best.pt"
            pre_test = {
                "protocol": "calibration_ready_pre_test",
                "testEvaluationCount": 0,
                "frozenThreshold": 0.57,
            }
            (root / "calibration_ready_summary.json").write_text(json.dumps(pre_test), encoding="utf-8")
            (root / "eval_summary.json").write_text(
                json.dumps(
                    {
                        "protocol": "frozen_calibration_one_shot_test",
                        "testEvaluationCount": 1,
                        "frozenThreshold": 0.91,
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                checkpoint=checkpoint,
                eval_summary=None,
                split="validation",
                threshold=None,
                allow_test_diagnostic=False,
            )
            threshold, source = interventions.load_threshold(args)
            self.assertEqual(threshold, 0.57)
            self.assertEqual(source["protocol"], "calibration_ready_pre_test")

            args.split = "test"
            threshold, source = interventions.load_threshold(args)
            self.assertEqual(threshold, 0.57)
            self.assertEqual(source["protocol"], "calibration_ready_pre_test")

            (root / "calibration_ready_summary.json").unlink()
            with self.assertRaisesRegex(ValueError, "only be used with"):
                interventions.load_threshold(args)
            args.allow_test_diagnostic = True
            threshold, source = interventions.load_threshold(args)
            self.assertEqual(threshold, 0.91)
            self.assertEqual(source["protocol"], "frozen_calibration_one_shot_test")


if __name__ == "__main__":
    unittest.main()
