from __future__ import annotations

from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest import mock

BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "benchmark"
MODEL_DIR = Path(__file__).resolve().parents[1]
for path in (str(BENCHMARK_DIR), str(MODEL_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from run_pvs import MAINLINE_CONFIG, build_train_command  # noqa: E402
from fixed_geometry_encoder import InstancePointNetPPGeoEncoder  # noqa: E402
from train_pvs import (  # noqa: E402
    _calibration_blend,
    _initialize_model_from_checkpoint,
    _load_relation_bundle,
    _resolve_split,
    _validation_key,
    _validate_args,
    _weighted_recall_safety_gate,
    parse_args,
)

DATA_ROOT = Path(os.environ.get("SLM_DATA_ROOT", "/mnt/sda/rhyang/slm")).resolve()


class TrainPvsTests(unittest.TestCase):
    def test_offline_geometry_encoder_output_shape(self) -> None:
        import torch

        encoder = InstancePointNetPPGeoEncoder(
            point_feature_dim=16,
            hidden_dim=8,
            out_dim=96,
            center_count=4,
            neighbor_count=3,
        )
        output = encoder(torch.randn(2, 8, 16))
        self.assertEqual(tuple(output.shape), (2, 96))
        self.assertTrue(bool(torch.isfinite(output).all()))

    def test_relation_tensor_bundle_preserves_current_schema(self) -> None:
        relation_dir = (
            DATA_ROOT
            / "neural_instance_culling/dataset/out/pvs_v4_integrated_visibility_mainline_v1"
            / "bounded_relation_csr_v3"
        )
        relation, tensors, *_rest = _load_relation_bundle(relation_dir, 18831)
        self.assertEqual(tensors["metadata"]["schema"], relation.metadata["schema"])
        self.assertTrue(tensors["metadata"]["trainOnly"])

    def test_mainline_runner_arguments_match_the_only_training_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            command = build_train_command(
                Path("."),
                Path(temporary) / "member",
                MAINLINE_CONFIG,
                "full",
                seed=20260801,
                smoke=True,
            )
            args = parse_args(command[2:])
        self.assertEqual(args.loss_variant, "pose_balanced_rvl_contrastive")
        self.assertEqual(args.occlusion_representation, "survival")
        self.assertEqual(args.relation_source, "bounded_hierarchical")
        self.assertEqual(args.spectral_mode, "moment_envelope")
        self.assertEqual(args.seed, 20260801)
        _validate_args(args)

    def test_test_split_is_rejected_before_dataset_access(self) -> None:
        with self.assertRaisesRegex(ValueError, "test is forbidden"):
            _resolve_split(mock.Mock(), "test", "validation")

    def test_validation_safety_requires_weighted_recall_and_lcb(self) -> None:
        self.assertTrue(
            _weighted_recall_safety_gate(
                {
                    "aggregateWeightedRecall": 0.995,
                    "aggregateWeightedRecallLowerConfidenceBound": 0.991,
                }
            )
        )
        self.assertFalse(
            _weighted_recall_safety_gate(
                {
                    "aggregateWeightedRecall": 0.999,
                    "aggregateWeightedRecallLowerConfidenceBound": 0.99,
                }
            )
        )
        self.assertFalse(
            _weighted_recall_safety_gate(
                {
                    "aggregateWeightedRecall": 0.99,
                    "aggregateWeightedRecallLowerConfidenceBound": 0.999,
                }
            )
        )

    def test_safe_checkpoint_ranking_uses_validation_first(self) -> None:
        calibration = {"agg_useful_cull": 0.9}
        lower_precision = {
            "agg_balanced_accuracy": 0.8,
            "agg_precision": 0.4,
            "agg_accuracy": 0.8,
            "agg_useful_cull": 0.7,
            "avg_pred_count": 10,
        }
        higher_precision = dict(lower_precision, agg_precision=0.5)
        self.assertGreater(
            _validation_key(higher_precision, calibration),
            _validation_key(lower_precision, calibration),
        )

    def test_instance_calibration_schedule_warms_up_and_reaches_one(self) -> None:
        self.assertEqual(_calibration_blend(0, 100, 0.1, 0.2), 0.0)
        self.assertGreater(_calibration_blend(20, 100, 0.1, 0.2), 0.0)
        self.assertEqual(_calibration_blend(99, 100, 0.1, 0.2), 1.0)

    def test_warm_start_loads_v4_model_state_but_records_fresh_adamw(self) -> None:
        import torch

        from pvs_model import BoundedRelationSurvivalMomentModel
        from train_pvs import CHECKPOINT_SCHEMA, MODEL_SCHEMA, TRAINING_SCHEMA

        source = BoundedRelationSurvivalMomentModel(
            2,
            1,
            survival_rank=4,
            relation_source="bounded_hierarchical",
            occlusion_representation="survival",
            spectral_mode="moment_envelope",
            depth_q01=1.0,
            depth_q99=2.0,
            depth_epsilon=1e-6,
            instance_calibration_mode="residual",
        )
        source.set_instance_calibration_blend(0.73)
        checkpoint_config = dict(source.config)
        checkpoint_config["occlusionRepresentation"] = dict(
            checkpoint_config["occlusionRepresentation"],
            querySemantics="monotone_logistic_survival",
            offlineSupervision="train_only_relation_and_depth_censoring",
        )
        checkpoint_config["queryTailSeparator"] = {"enabled": False}
        checkpoint = {
            "schema": CHECKPOINT_SCHEMA,
            "runtimeSchema": MODEL_SCHEMA,
            "modelConfig": checkpoint_config,
            "modelState": source.state_dict(),
            "protocol": {
                "schema": TRAINING_SCHEMA,
                "seed": 20260802,
                "dataset": {"path": "/data/ifcbench"},
                "runtimeMeta": {"path": "/data/ifcbench/runtime.json"},
                "viewcell": {"shape": "horizontal_disk", "radiusM": 2.5},
                "testRead": False,
            },
            "experimentName": "source_v4",
            "epoch": 40,
            "globalStep": 36000,
            "testRead": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.pt"
            torch.save(checkpoint, path)
            target = BoundedRelationSurvivalMomentModel(
                2,
                1,
                survival_rank=4,
                relation_source="bounded_hierarchical",
                occlusion_representation="survival",
                spectral_mode="moment_envelope",
                depth_q01=1.0,
                depth_q99=2.0,
                depth_epsilon=1e-6,
                instance_calibration_mode="residual",
            )
            initialization = _initialize_model_from_checkpoint(target, path)
        self.assertEqual(initialization["mode"], "from-checkpoint")
        self.assertEqual(initialization["sourceSeed"], 20260802)
        self.assertEqual(initialization["sourceEpoch"], 40)
        self.assertEqual(initialization["sourceGlobalStep"], 36000)
        self.assertFalse(initialization["optimizerStateLoaded"])
        self.assertEqual(initialization["optimizerStateSource"], "new")
        self.assertAlmostEqual(initialization["inheritedInstanceCalibrationBlend"], 0.73)
        self.assertEqual(
            initialization["inheritedProtocol"]["dataset"]["path"],
            "/data/ifcbench",
        )
        self.assertEqual(initialization["inheritedViewcell"]["radiusM"], 2.5)
        self.assertAlmostEqual(float(target.instance_calibration_blend), 0.73)


if __name__ == "__main__":
    unittest.main()
