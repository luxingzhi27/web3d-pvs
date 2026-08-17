from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import unittest

BENCHMARK = Path(__file__).resolve().parents[1]
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))

import evaluate_pvs_bounded_relation_survival_moment_v4 as evaluator  # noqa: E402


class BoundedRelationSurvivalMomentV3EvaluatorTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
