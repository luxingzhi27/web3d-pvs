from __future__ import annotations

import argparse
import copy
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
            ("train", "calibration", "validation", "test"),
        )
        self.assertIn("test", evaluator.REPLAY_SPLITS)

    def test_formal_test_requires_explicit_calibration_and_rejects_selection_flags(self) -> None:
        base = argparse.Namespace(
            split="test",
            calibration=None,
            allow_unsafe_diagnostic=False,
            diagnostic_recalibrate=False,
        )
        with self.assertRaisesRegex(ValueError, "explicit --calibration"):
            evaluator.validate_evaluation_mode(base)

        base.calibration = Path("calibration_ready_summary.json")
        base.allow_unsafe_diagnostic = True
        with self.assertRaisesRegex(ValueError, "unsafe"):
            evaluator.validate_evaluation_mode(base)

        base.allow_unsafe_diagnostic = False
        base.diagnostic_recalibrate = True
        with self.assertRaisesRegex(ValueError, "recalibration"):
            evaluator.validate_evaluation_mode(base)

    def test_validation_mode_remains_test_free(self) -> None:
        args = argparse.Namespace(
            split="validation",
            calibration=None,
            allow_unsafe_diagnostic=False,
            diagnostic_recalibrate=False,
        )
        self.assertFalse(evaluator.validate_evaluation_mode(args))

    def test_legacy_fingerprint_filter_preserves_shape_fields(self) -> None:
        value = evaluator._strip_fingerprint_fields(
            {
                "shape": [18831, 96],
                "dtype": "float16",
                "geometrySha256": "legacy",
                "artifactFiles": {"weights.bin": "legacy"},
            }
        )
        self.assertEqual(value, {"shape": [18831, 96], "dtype": "float16"})

    def test_json_safe_metadata_represents_open_sampler_bounds_as_null(self) -> None:
        value = evaluator._json_safe_metadata({"depthEdges": [-math.inf, 0.5, math.inf]})
        self.assertEqual(value, {"depthEdges": [None, 0.5, None]})
        self.assertIs(evaluator._json_safe_metadata({"testRead": False})["testRead"], False)

    def test_model_meta_provenance_is_strict_after_json_scalar_normalization(self) -> None:
        config = {
            "runtimeSchema": "runtime-v4",
            "runtimeFeatureDim": 124,
            "frequency": {"count": 16, "maxNormCycles": 8.0},
        }
        checkpoint_protocol = {
            "schema": "training-v4",
            "experiment": "registered-experiment-name",
            "lossVariant": "pose_balanced_rvl_contrastive",
            "splitPoseCounts": {"train": 2, "calibration": 1, "validation": 1},
            "testRead": False,
            "legacySha256": "ignored",
        }
        model_meta = {
            "modelConfig": {
                "runtimeSchema": "runtime-v4",
                "runtimeFeatureDim": 124.0,
                "frequency": {"count": 16.0, "maxNormCycles": 8},
            },
            "protocol": {
                "schema": "training-v4",
                "experiment": "/output/run-directory",
                "lossVariant": "pose_balanced_rvl_contrastive",
                "splitPoseCounts": {"train": 2.0, "calibration": 1.0, "validation": 1.0},
                "testRead": False,
            },
        }
        evaluator._validate_model_meta_provenance(
            model_meta, config, checkpoint_protocol
        )

        changed = copy.deepcopy(model_meta)
        changed["modelConfig"]["runtimeFeatureDim"] = 130
        with self.assertRaisesRegex(ValueError, "modelConfig"):
            evaluator._validate_model_meta_provenance(
                changed, config, checkpoint_protocol
            )

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

    def test_frozen_threshold_source_drops_pose_arrays(self) -> None:
        selection = {
            "threshold": 0.125,
            "aggregateWeightedRecall": 0.996,
            "aggregateWeightedRecallLowerConfidenceBound": 0.995,
            "_pose_weighted_recall_values": [0.99] * 100,
        }
        summary = {
            "schema": evaluator.CALIBRATION_SCHEMA,
            "status": "safe",
            "bestSafe": {"selection": selection},
            "calibration": {"thresholdRows": [{"threshold": 0.125}]},
            "testRead": False,
        }
        checkpoint = {"best": {"selection": selection}}
        _threshold, source = evaluator._frozen_threshold(
            checkpoint, summary, allow_unsafe=False
        )
        self.assertEqual(
            source["selection"],
            {
                "threshold": 0.125,
                "aggregateWeightedRecall": 0.996,
                "aggregateWeightedRecallLowerConfidenceBound": 0.995,
            },
        )

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

    def test_exact_calibration_is_checkpoint_specific_and_safe(self) -> None:
        checkpoint = Path("/tmp/member/last.pt")
        selected = {
            "threshold": 0.5945902,
            "aggregateWeightedRecall": 0.99043,
            "aggregateWeightedRecallLowerConfidenceBound": 0.99001,
            "agg_precision": 0.30,
        }
        summary = {
            "schema": evaluator.EXACT_CALIBRATION_SCHEMA,
            "split": "calibration",
            "checkpoint": str(checkpoint),
            "predictionRule": "score >= threshold",
            "status": "safe",
            "selection": {"threshold": selected["threshold"]},
            "selected": selected,
            "testRead": False,
        }
        threshold, source = evaluator._exact_frozen_threshold(checkpoint, summary)
        self.assertAlmostEqual(threshold, selected["threshold"])
        self.assertEqual(source["protocol"], "checkpoint_specific_exact_calibration")
        self.assertTrue(source["safeWorkpoint"])

        wrong = copy.deepcopy(summary)
        wrong["checkpoint"] = "/tmp/member/other.pt"
        with self.assertRaisesRegex(ValueError, "different checkpoint"):
            evaluator._exact_frozen_threshold(checkpoint, wrong)

        unsafe = copy.deepcopy(summary)
        unsafe["selected"]["aggregateWeightedRecallLowerConfidenceBound"] = 0.99
        with self.assertRaisesRegex(ValueError, "safety gate"):
            evaluator._exact_frozen_threshold(checkpoint, unsafe)

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
