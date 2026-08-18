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

import evaluate_pvs_bounded_relation_survival_moment_v4 as evaluator  # noqa: E402


class BoundedRelationSurvivalMomentV3EvaluatorTests(unittest.TestCase):
    def test_query_tail_separator_constructor_values_follow_checkpoint_schema(self) -> None:
        self.assertEqual(
            evaluator._query_tail_separator_constructor_values({}),
            ("disabled", 8, 0.5, "pose_mean"),
        )
        self.assertEqual(
            evaluator._query_tail_separator_constructor_values(
                {
                    "queryTailSeparator": {
                        "enabled": True,
                        "family": "mlp",
                        "hiddenDim": 8,
                        "maximumAbsoluteResidual": 0.5,
                        "centering": "pose_mean",
                    }
                }
            ),
            ("mlp", 8, 0.5, "pose_mean"),
        )
        with self.assertRaisesRegex(ValueError, "queryTailSeparator"):
            evaluator._query_tail_separator_constructor_values(
                {
                    "queryTailSeparator": {
                        "enabled": False,
                        "family": "linear",
                    }
                }
            )

    def test_replay_splits_allow_train_diagnostics_but_never_test(self) -> None:
        self.assertEqual(
            evaluator.REPLAY_SPLITS,
            ("train", "calibration", "validation"),
        )
        self.assertNotIn("test", evaluator.REPLAY_SPLITS)

    def test_checkpoint_reconstruction_passes_non_default_widths_and_region_head(self) -> None:
        captured: dict[str, object] = {}

        class StopAfterConstruction(RuntimeError):
            pass

        class CapturingModel:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

            def to(self, _device: object) -> "CapturingModel":
                return self

            def set_instance_world_aabbs(self, _value: object) -> None:
                return None

            def set_instance_to_glb(self, _value: object) -> None:
                return None

            def load_state_dict(self, _state: object, strict: bool = True) -> None:
                self.strict = strict
                raise StopAfterConstruction

        config = {
            "numInstances": 2,
            "numGlbs": 1,
            "runtimeFeatureDim": 124,
            "relationHiddenDim": 11,
            "hiddenDim": 13,
            "relationSource": "bounded_hierarchical",
            "spectralMode": "moment_extrema",
            "depthNormalization": {"q01": 0.0, "q99": 1.0, "epsilon": 1e-6},
            "frequency": {"maxNormCycles": 8.0},
            "instanceCalibration": {
                "mode": "residual",
                "maximumAbsoluteResidual": 4.0,
                "sparseInstancePenalty": 3.0,
            },
            "viewcellRegionConditionedVisibility": {
                "enabled": True,
                "regionInputDim": 27,
                "queryAuxKey": "viewcell_extreme_features",
                "queryAuxFeatureDim": 17,
                "hiddenInputDim": 13,
                "projectionDim": 16,
                "fusionDim": 48,
                "headHiddenDim": 16,
                "activation": "SiLU",
                "fusion": (
                    "concat(region_projection, hidden_projection, "
                    "region_projection * hidden_projection)"
                ),
                "output": "unbounded additive main visibility logit",
                "outputInitialization": "zero weight and bias",
                "centering": "pose_mean",
                "poseReduction": "candidate_mean_per_pose",
                "runtimeReduction": "candidate_mean_per_pose",
                "boundedCorrection": False,
            },
        }
        checkpoint = {
            "schema": evaluator.CHECKPOINT_SCHEMA,
            "runtimeSchema": evaluator.MODEL_SCHEMA,
            "testRead": False,
            "protocol": {
                "schema": evaluator.TRAINING_SCHEMA,
                "testRead": False,
            },
            "modelConfig": config,
            "modelState": {},
        }
        world_aabbs = np.zeros((2, 6), dtype=np.float32)
        instance_to_glb = np.zeros((2,), dtype=np.int64)
        args = argparse.Namespace(runtime_meta="runtime.json", device="cpu")
        with (
            mock.patch.object(
                evaluator,
                "load_runtime_meta",
                return_value=(world_aabbs, instance_to_glb, {}),
            ),
            mock.patch.object(
                evaluator, "BoundedRelationSurvivalMomentModel", CapturingModel
            ),
            self.assertRaises(StopAfterConstruction),
        ):
            evaluator._evaluate_checkpoint(args, checkpoint, Path("checkpoint.pt"))

        self.assertEqual(captured["relation_hidden_dim"], 11)
        self.assertEqual(captured["hidden_dim"], 13)
        self.assertTrue(captured["viewcell_region_conditioned_visibility_enabled"])
        self.assertEqual(
            captured["viewcell_region_conditioned_visibility_centering"],
            "pose_mean",
        )

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

    def test_view_residual_constructor_values_follow_checkpoint_schema(self) -> None:
        self.assertEqual(
            evaluator._view_residual_constructor_values({}),
            (0.0, 16),
        )
        self.assertEqual(
            evaluator._view_residual_constructor_values(
                {
                    "viewResidual": {
                        "enabled": True,
                        "maximumAbsoluteResidual": 1.5,
                        "hiddenDim": 24,
                    }
                }
            ),
            (1.5, 24),
        )
        for residual in (
            "invalid",
            {"enabled": True},
            {"enabled": True, "maximumAbsoluteResidual": 0.0, "hiddenDim": 16},
        ):
            with self.assertRaisesRegex(ValueError, "viewResidual"):
                evaluator._view_residual_constructor_values(
                    {"viewResidual": residual}
                )

    def test_boundary_opportunity_constructor_values_follow_checkpoint_schema(self) -> None:
        self.assertEqual(
            evaluator._boundary_opportunity_constructor_values({}),
            (0, 24, -6.0, 6.0),
        )
        self.assertEqual(
            evaluator._boundary_opportunity_constructor_values(
                {
                    "boundaryOpportunity": {
                        "enabled": True,
                        "hiddenDim": 32,
                        "projectionDim": 24,
                        "initialLogit": -6.0,
                        "maximumLogitUplift": 8.0,
                    }
                }
            ),
            (32, 24, -6.0, 8.0),
        )
        for opportunity in (
            "invalid",
            {"enabled": True},
            {
                "enabled": True,
                "hiddenDim": 0,
                "projectionDim": 24,
                "initialLogit": -6.0,
                "maximumLogitUplift": 6.0,
            },
        ):
            with self.assertRaisesRegex(ValueError, "boundaryOpportunity"):
                evaluator._boundary_opportunity_constructor_values(
                    {"boundaryOpportunity": opportunity}
                )

    def test_boundary_tail_residual_constructor_values_follow_checkpoint_schema(self) -> None:
        self.assertEqual(
            evaluator._boundary_tail_residual_constructor_values({}),
            (0, 24, 1.0, "pose_mean", "none", "product", 0.0),
        )
        self.assertEqual(
            evaluator._boundary_tail_residual_constructor_values(
                {
                    "boundaryTailResidual": {
                        "enabled": True,
                        "hiddenDim": 32,
                        "projectionDim": 20,
                        "maximumAbsoluteResidual": 1.5,
                        "centering": "none",
                        "shortcut": "region_linear",
                        "fusionMode": "affine_region",
                        "outputInitializationStd": 0.002,
                    }
                }
            ),
            (32, 20, 1.5, "none", "region_linear", "affine_region", 0.002),
        )
        for residual in (
            "invalid",
            {"enabled": True},
            {
                "enabled": True,
                "hiddenDim": 0,
                "projectionDim": 24,
                "maximumAbsoluteResidual": 1.0,
            },
            {
                "enabled": True,
                "hiddenDim": 16,
                "projectionDim": 24,
                "maximumAbsoluteResidual": 1.0,
                "centering": "invalid",
            },
            {
                "enabled": True,
                "hiddenDim": 16,
                "projectionDim": 24,
                "maximumAbsoluteResidual": 1.0,
                "shortcut": "invalid",
            },
        ):
            with self.assertRaisesRegex(ValueError, "boundaryTailResidual"):
                evaluator._boundary_tail_residual_constructor_values(
                    {"boundaryTailResidual": residual}
                )

    def test_viewcell_extreme_visibility_constructor_defaults_keep_old_checkpoint_compatible(self) -> None:
        self.assertFalse(
            evaluator._viewcell_extreme_visibility_constructor_value({})
        )
        self.assertFalse(
            evaluator._viewcell_extreme_visibility_constructor_value(
                {"viewcellExtremeVisibility": {"enabled": False}}
            )
        )

    def test_viewcell_extreme_visibility_constructor_accepts_enabled_schema(self) -> None:
        self.assertTrue(
            evaluator._viewcell_extreme_visibility_constructor_value(
                {
                    "viewcellExtremeVisibility": {
                        "enabled": True,
                        "inputDim": 27,
                        "queryAuxKey": "viewcell_extreme_features",
                        "queryAuxFeatureDim": 17,
                        "projectionDim": 16,
                        "activation": "SiLU",
                        "poseReduction": "none",
                        "boundedCorrection": False,
                    }
                }
            )
        )

    def test_viewcell_extreme_visibility_constructor_rejects_schema_drift(self) -> None:
        base = {
            "enabled": True,
            "inputDim": 27,
            "queryAuxKey": "viewcell_extreme_features",
            "queryAuxFeatureDim": 17,
            "projectionDim": 16,
            "activation": "SiLU",
            "poseReduction": "none",
            "boundedCorrection": False,
        }
        for field, value in (
            ("inputDim", 26),
            ("queryAuxKey", "other_features"),
            ("queryAuxFeatureDim", 16),
            ("projectionDim", 8),
            ("activation", "ReLU"),
            ("poseReduction", "pose_mean"),
            ("boundedCorrection", True),
        ):
            with self.subTest(field=field):
                config = dict(base)
                config[field] = value
                with self.assertRaisesRegex(ValueError, "viewcellExtremeVisibility"):
                    evaluator._viewcell_extreme_visibility_constructor_value(
                        {"viewcellExtremeVisibility": config}
                    )
        with self.assertRaisesRegex(ValueError, "viewcellExtremeVisibility"):
            evaluator._viewcell_extreme_visibility_constructor_value(
                {"viewcellExtremeVisibility": "invalid"}
            )

    def test_region_conditioned_visibility_constructor_defaults_disabled(self) -> None:
        self.assertEqual(
            evaluator._viewcell_region_conditioned_visibility_constructor_value({}),
            (False, "none"),
        )
        self.assertEqual(
            evaluator._viewcell_region_conditioned_visibility_constructor_value(
                {"viewcellRegionConditionedVisibility": {"enabled": False}}
            ),
            (False, "none"),
        )

    def test_region_conditioned_visibility_constructor_accepts_enabled_schema(self) -> None:
        self.assertEqual(
            evaluator._viewcell_region_conditioned_visibility_constructor_value(
                {
                    "hiddenDim": 64,
                    "viewcellRegionConditionedVisibility": {
                        "enabled": True,
                        "regionInputDim": 27,
                        "queryAuxKey": "viewcell_extreme_features",
                        "queryAuxFeatureDim": 17,
                        "hiddenInputDim": 64,
                        "projectionDim": 16,
                        "fusionDim": 48,
                        "headHiddenDim": 16,
                        "activation": "SiLU",
                        "fusion": (
                            "concat(region_projection, hidden_projection, "
                            "region_projection * hidden_projection)"
                        ),
                        "output": "unbounded additive main visibility logit",
                        "outputInitialization": "zero weight and bias",
                        "centering": "pose_mean",
                        "poseReduction": "candidate_mean_per_pose",
                        "runtimeReduction": "candidate_mean_per_pose",
                        "boundedCorrection": False,
                    },
                }
            ),
            (True, "pose_mean"),
        )

    def test_region_conditioned_visibility_constructor_accepts_uncentered_schema(self) -> None:
        config = {
            "enabled": True,
            "regionInputDim": 27,
            "queryAuxKey": "viewcell_extreme_features",
            "queryAuxFeatureDim": 17,
            "hiddenInputDim": 64,
            "projectionDim": 16,
            "fusionDim": 48,
            "headHiddenDim": 16,
            "activation": "SiLU",
            "fusion": (
                "concat(region_projection, hidden_projection, "
                "region_projection * hidden_projection)"
            ),
            "output": "unbounded additive main visibility logit",
            "outputInitialization": "zero weight and bias",
            "centering": "none",
            "poseReduction": "none",
            "runtimeReduction": "none",
            "boundedCorrection": False,
        }
        self.assertEqual(
            evaluator._viewcell_region_conditioned_visibility_constructor_value(
                {"viewcellRegionConditionedVisibility": config}
            ),
            (True, "none"),
        )

    def test_region_conditioned_visibility_constructor_rejects_schema_drift(self) -> None:
        base = {
            "enabled": True,
            "regionInputDim": 27,
            "queryAuxKey": "viewcell_extreme_features",
            "queryAuxFeatureDim": 17,
            "hiddenInputDim": 64,
            "projectionDim": 16,
            "fusionDim": 48,
            "headHiddenDim": 16,
            "activation": "SiLU",
            "fusion": (
                "concat(region_projection, hidden_projection, "
                "region_projection * hidden_projection)"
            ),
            "output": "unbounded additive main visibility logit",
            "outputInitialization": "zero weight and bias",
            "centering": "pose_mean",
            "poseReduction": "candidate_mean_per_pose",
            "runtimeReduction": "candidate_mean_per_pose",
            "boundedCorrection": False,
        }
        for field, value in (
            ("regionInputDim", 26),
            ("queryAuxKey", "other_features"),
            ("queryAuxFeatureDim", 16),
            ("hiddenInputDim", 32),
            ("projectionDim", 8),
            ("fusionDim", 32),
            ("headHiddenDim", 8),
            ("activation", "ReLU"),
            ("fusion", "add"),
            ("centering", "unknown"),
            ("poseReduction", "none"),
            ("runtimeReduction", "none"),
            ("boundedCorrection", True),
        ):
            with self.subTest(field=field):
                config = dict(base)
                config[field] = value
                with self.assertRaisesRegex(
                    ValueError, "viewcellRegionConditionedVisibility"
                ):
                    evaluator._viewcell_region_conditioned_visibility_constructor_value(
                        {"viewcellRegionConditionedVisibility": config}
                    )
        with self.assertRaisesRegex(ValueError, "viewcellRegionConditionedVisibility"):
            evaluator._viewcell_region_conditioned_visibility_constructor_value(
                {"viewcellRegionConditionedVisibility": "invalid"}
            )

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
