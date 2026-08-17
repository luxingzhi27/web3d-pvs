from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


BENCHMARK = Path(__file__).resolve().parents[1]
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))

import validate_pvs_bounded_relation_survival_moment_v4 as validator  # noqa: E402


class BoundedRelationSurvivalMomentV4ValidatorTests(unittest.TestCase):
    @staticmethod
    def _fixture() -> tuple[dict, dict, dict]:
        digest = lambda char: char * 64
        config = {
            "runtimeSchema": validator.RUNTIME_SCHEMA,
            "numInstances": 2,
            "numGlbs": 1,
            "relationSource": "bounded_hierarchical",
            "spectralMode": "moment_envelope",
            "geometryDim": 96,
            "survivalCoefficientShape": [4, 7],
            "runtimeFeatureDim": 124,
            "instanceCalibration": {
                "mode": "residual",
                "shape": [4, 7],
                "maximumAbsoluteResidual": 4.0,
                "sparseInstancePenalty": 3.0,
                "runtimeExport": "fused coefficients only",
            },
            "depthNormalization": {"q01": 0.0, "q99": 1.0, "epsilon": 1e-6, "sourceSplit": "train"},
            "frequency": {"count": 16, "units": "cycles", "maxNormCycles": 8.0},
        }
        protocol = {
            "schema": validator.TRAINING_SCHEMA,
            "variant": "full",
            "lossVariant": "safety_reserve",
            "seed": 20260801,
            "testRead": False,
            "thresholdSource": "checkpoint-own-calibration-only",
            "candidateUnion": False,
            "splitNames": {"train": "train", "calibration": "calibration", "validation": "validation"},
            "splitPoseCounts": dict(validator.EXPECTED_SPLIT_POSE_COUNTS),
            "candidateDigests": {"train": digest("a"), "calibration": digest("b"), "validation": digest("c")},
            "viewcell": {
                "shape": "horizontal_disk",
                "radiusM": 2.0,
                "candidateCameraSemantics": "66-degree back-camera candidate identity only",
                "queryCenterSemantics": "center of the same-direction view-cell visibility union",
            },
            "instanceCalibration": {"mode": "residual"},
        }
        selection = {
            "threshold": 0.25,
            "aggregateWeightedRecall": 0.995,
            "aggregateWeightedRecallLowerConfidenceBound": 0.994,
            "eval_pose_count": validator.EXPECTED_SPLIT_POSE_COUNTS["calibration"],
        }
        calibration_payload = {
            "schema": validator.CALIBRATION_ROW_SCHEMA,
            "thresholdSource": "this-checkpoint-calibration-only",
            "selectedSafe": selection,
            "diagnostic": selection,
            "thresholdRows": [{"threshold": 0.25}],
            "testRead": False,
        }
        calibration = {
            "schema": validator.CALIBRATION_SUMMARY_SCHEMA,
            "status": "safe",
            "bestSafe": {"threshold": 0.25, "safe": True, "selection": selection},
            "bestDiagnostic": {"threshold": 0.25, "safe": True, "selection": selection},
            "calibration": calibration_payload,
            "validationAtFrozenThreshold": {
                "eval_pose_count": validator.EXPECTED_SPLIT_POSE_COUNTS["validation"],
            },
            "testRead": False,
        }
        checkpoint = {
            "schema": validator.CHECKPOINT_SCHEMA,
            "runtimeSchema": validator.RUNTIME_SCHEMA,
            "protocol": protocol,
            "modelConfig": config,
            "modelState": {
                "instance_calibration_residual_raw": torch.zeros(2, 4, 7),
            },
            "instanceSurvivalCoefficients": torch.zeros(2, 4, 7),
            "instanceSurvivalPriorCoefficients": torch.zeros(2, 4, 7),
            "instanceSurvivalCalibrationResidual": torch.zeros(2, 4, 7),
            "instanceCalibration": {
                "mode": "residual",
                "blend": 1.0,
                "fusion": "prior_plus_applied_residual",
                "runtimeExport": "fused_coefficients_only",
            },
            "relation": {
                "schema": validator.RELATION_SCHEMA,
                "edgeCount": 4,
                "rowCount": 5,
                "artifactDigest": digest("d"),
                "candidateDigest": digest("a"),
                "hierarchy": {"localGroupCount": 1, "structuralGroupCount": 1},
                "observations": {
                    "observationCount": 6,
                    "eventCount": 4,
                    "rightCensoredCount": 2,
                },
            },
            "geometry": {"sha256": digest("e"), "shape": [2, 96], "dtype": "float16"},
            "calibration": calibration_payload,
            "best": {"threshold": 0.25, "safe": True, "selection": selection},
            "epoch": 1,
            "testRead": False,
        }
        model_meta = {
            "schema": validator.MODEL_SCHEMA,
            "modelConfig": config,
            "protocol": protocol,
            "runtimeFeature": {"shape": [2, 124], "dtype": "float16"},
            "testRead": False,
        }
        return checkpoint, model_meta, calibration

    def test_v4_artifact_uses_registered_variant_and_calibration(self) -> None:
        checkpoint, model_meta, calibration = self._fixture()
        result = validator.validate_v4_training_artifact(
            checkpoint,
            model_meta=model_meta,
            calibration_summary=calibration,
        )
        self.assertEqual(result["variant"], "full")
        self.assertEqual(result["variantSpec"]["relationArtifact"], "native")
        self.assertEqual(result["threshold"], 0.25)
        self.assertFalse(result["testRead"])

    def test_v4_validator_rejects_historical_checkpoint(self) -> None:
        with self.assertRaises(ValueError):
            validator.validate_v4_training_artifact({"schema": "pvs-hierarchical-relation-survival-integrated-training-v1"})

    def test_v4_validator_rejects_unregistered_variant(self) -> None:
        checkpoint, model_meta, calibration = self._fixture()
        checkpoint["protocol"]["variant"] = "historical_strong_reference"
        with self.assertRaisesRegex(ValueError, "unknown v4 variant"):
            validator.validate_v4_training_artifact(checkpoint, model_meta=model_meta, calibration_summary=calibration)

    def test_v4_validator_rejects_loss_variant_mismatch(self) -> None:
        checkpoint, model_meta, calibration = self._fixture()
        checkpoint["protocol"]["lossVariant"] = "normalized_rvl"
        with self.assertRaisesRegex(ValueError, "loss variant"):
            validator.validate_v4_training_artifact(
                checkpoint,
                model_meta=model_meta,
                calibration_summary=calibration,
            )

    def test_v4_validator_rejects_nonzero_disabled_residual(self) -> None:
        checkpoint, model_meta, calibration = self._fixture()
        checkpoint["protocol"]["variant"] = "without_instance_calibration_residual"
        checkpoint["protocol"]["instanceCalibration"]["mode"] = "disabled"
        checkpoint["modelConfig"]["instanceCalibration"]["mode"] = "disabled"
        checkpoint["instanceCalibration"]["mode"] = "disabled"
        checkpoint["instanceCalibration"]["blend"] = 0.0
        checkpoint["modelState"].pop("instance_calibration_residual_raw")
        checkpoint["instanceSurvivalCalibrationResidual"] = torch.full((2, 4, 7), 0.25)
        checkpoint["instanceSurvivalCoefficients"] = torch.full((2, 4, 7), 0.25)
        with self.assertRaisesRegex(ValueError, "non-zero residual"):
            validator.validate_v4_training_artifact(
                checkpoint,
                model_meta=model_meta,
                calibration_summary=calibration,
            )

    def test_v4_validator_accepts_registered_fp16_fusion_rounding(self) -> None:
        checkpoint, model_meta, calibration = self._fixture()
        prior = torch.full((2, 4, 7), 8.0, dtype=torch.float16)
        residual = torch.full((2, 4, 7), 0.75, dtype=torch.float16)
        fused = torch.full((2, 4, 7), 8.75, dtype=torch.float16)
        checkpoint["instanceSurvivalPriorCoefficients"] = prior
        checkpoint["instanceSurvivalCalibrationResidual"] = residual
        checkpoint["instanceSurvivalCoefficients"] = fused
        result = validator.validate_v4_training_artifact(
            checkpoint,
            model_meta=model_meta,
            calibration_summary=calibration,
        )
        self.assertEqual(result["variant"], "full")

    def test_v4_validator_rejects_unregistered_frequency_range(self) -> None:
        checkpoint, model_meta, calibration = self._fixture()
        checkpoint["modelConfig"]["frequency"]["maxNormCycles"] = 8.01
        with self.assertRaisesRegex(ValueError, "frequency norm"):
            validator.validate_v4_training_artifact(
                checkpoint,
                model_meta=model_meta,
                calibration_summary=calibration,
            )

    def test_v4_calibration_tamper_is_rejected(self) -> None:
        checkpoint, model_meta, calibration = self._fixture()
        calibration["bestSafe"]["selection"]["threshold"] = 0.5
        with self.assertRaises(ValueError):
            validator.validate_v4_training_artifact(checkpoint, model_meta=model_meta, calibration_summary=calibration)

    def test_v4_self_test_is_asset_free(self) -> None:
        result = validator.self_test()
        self.assertEqual(result, {"status": "passed", "schema": validator.CHECKPOINT_SCHEMA})


if __name__ == "__main__":
    unittest.main()
