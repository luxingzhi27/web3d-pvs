from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

BENCHMARK = Path(__file__).resolve().parents[1]
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))

import evaluate_pvs as evaluator  # noqa: E402


class PvsV4EvaluatorTests(unittest.TestCase):
    def test_replay_splits_allow_train_diagnostics_but_never_test(self) -> None:
        self.assertEqual(
            evaluator.REPLAY_SPLITS,
            ("train", "calibration", "validation"),
        )
        self.assertNotIn("test", evaluator.REPLAY_SPLITS)

    def test_legacy_fingerprint_filter_preserves_shape_fields(self) -> None:
        value = evaluator._strip_fingerprint_fields(
            {"shape": [18831, 96], "dtype": "float16", "geometrySha256": "legacy"}
        )
        self.assertEqual(value, {"shape": [18831, 96], "dtype": "float16"})

    def test_json_safe_metadata_represents_open_sampler_bounds_as_null(self) -> None:
        value = evaluator._json_safe_metadata({"depthEdges": [-math.inf, 0.5, math.inf]})
        self.assertEqual(value, {"depthEdges": [None, 0.5, None]})

    def test_frequency_contract_requires_explicit_registered_upper_bound(self) -> None:
        self.assertEqual(
            evaluator._max_frequency_norm_cycles_from_config(
                {"frequency": {"maxNormCycles": 8.0}}
            ),
            8.0,
        )
        for config in (
            {},
            {"frequency": {}},
            {"frequency": {"maxNormCycles": 8.0001}},
            {"frequency": {"maxNormCycles": 0.0}},
        ):
            with self.assertRaisesRegex(ValueError, "maxNormCycles"):
                evaluator._max_frequency_norm_cycles_from_config(config)

    def test_validation_replay_consumes_checkpoint_calibration_without_selecting(self) -> None:
        selection = {
            "threshold": 0.125,
            "aggregateWeightedRecall": 0.996,
            "aggregateWeightedRecallLowerConfidenceBound": 0.995,
        }
        summary = {
            "schema": evaluator.CALIBRATION_SCHEMA,
            "status": "safe",
            "bestSafe": {"selection": selection},
            "calibration": {"thresholdRows": [{"threshold": 0.125}]},
            "testRead": False,
        }
        checkpoint = {"best": {"selection": {"threshold": 0.125}}}
        threshold, source = evaluator._frozen_threshold(checkpoint, summary, allow_unsafe=False)
        self.assertEqual(threshold, 0.125)
        self.assertEqual(source["selectionSplit"], "calibration")
        self.assertEqual(source["protocol"], "checkpoint_own_calibration_only")
        self.assertFalse(source["selectedFromTest"])
        self.assertEqual(source["testEvaluationCount"], 0)

    def test_validation_replay_does_not_promote_diagnostic_to_safe(self) -> None:
        selection = {"threshold": 0.2}
        summary = {
            "schema": evaluator.CALIBRATION_SCHEMA,
            "status": "no_qualified_safety_workpoint",
            "bestDiagnostic": {"selection": selection},
            "calibration": {"thresholdRows": [{"threshold": 0.2}]},
            "testRead": False,
        }
        checkpoint = {"best": {"selection": selection}}
        threshold, source = evaluator._frozen_threshold(checkpoint, summary, allow_unsafe=True)
        self.assertEqual(threshold, 0.2)
        self.assertFalse(source["safeWorkpoint"])
        with self.assertRaises(ValueError):
            evaluator._frozen_threshold(checkpoint, summary, allow_unsafe=False)

    def test_v4_evaluator_rejects_a_legacy_checkpoint_before_asset_access(self) -> None:
        args = argparse.Namespace()
        with self.assertRaisesRegex(ValueError, "non-v4 checkpoint"):
            evaluator._evaluate_checkpoint(args, {"schema": "pvs-hierarchical-relation-survival-integrated-v2"}, Path("checkpoint.pt"))

    def test_diagnostic_recalibration_freezes_calibration_threshold_for_validation(self) -> None:
        calibration_split = mock.Mock(pose_indices=np.arange(3, dtype=np.int64))
        validation_split = mock.Mock(pose_indices=np.arange(4, dtype=np.int64))
        dataset = mock.Mock()
        dataset.split.side_effect = lambda name: {
            "calibration": calibration_split,
            "validation": validation_split,
        }[name]
        selected = {
            "threshold": 0.0125,
            "aggregateWeightedRecall": 0.995,
            "aggregateWeightedRecallLowerConfidenceBound": 0.992,
            "agg_useful_cull": 0.8,
        }
        validation = {
            "threshold": 0.0125,
            "aggregateWeightedRecall": 0.994,
            "aggregateWeightedRecallLowerConfidenceBound": 0.991,
        }
        args = argparse.Namespace(
            recalibration_bootstrap_replicates=123,
            poses_per_batch=2,
            seed=20260801,
        )
        with (
            mock.patch.object(
                evaluator,
                "evaluate_thresholds",
                side_effect=[[selected], [validation]],
            ) as evaluate,
            mock.patch.object(
                evaluator,
                "select_aggregate_weighted_cull_workpoint",
                return_value=selected,
            ),
            mock.patch.object(
                evaluator,
                "threshold_grid",
                return_value=np.asarray([0.001, 0.0125], dtype=np.float32),
            ),
        ):
            payload = evaluator._diagnostic_recalibration(
                args,
                {"epoch": 8, "protocol": {"seed": 20260801}},
                Path("best_safe.pt"),
                mock.Mock(),
                dataset,
                mock.Mock(),
                np.zeros((1, 6), dtype=np.float32),
                np.zeros((1,), dtype=np.int64),
                np.ones((1,), dtype=np.float32),
                mock.Mock(),
            )
        self.assertEqual(evaluate.call_count, 2)
        calibration_call = evaluate.call_args_list[0]
        validation_call = evaluate.call_args_list[1]
        self.assertIs(calibration_call.args[1], calibration_split)
        self.assertEqual(calibration_call.kwargs["bootstrap_replicates"], 123)
        self.assertIs(validation_call.args[1], validation_split)
        self.assertEqual(validation_call.kwargs["bootstrap_replicates"], 123)
        self.assertEqual(validation_call.kwargs["thresholds"].tolist(), [np.float32(0.0125)])
        self.assertEqual(payload["selectedSafe"]["threshold"], 0.0125)
        self.assertEqual(
            payload["validationAtSelectedThreshold"][
                "aggregateWeightedRecallLowerConfidenceBound"
            ],
            0.991,
        )


if __name__ == "__main__":
    unittest.main()
