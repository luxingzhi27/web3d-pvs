from __future__ import annotations

from pathlib import Path
import sys
import unittest


BENCHMARK = Path(__file__).resolve().parents[1]
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))

from validate_pvs_hierarchical_relation_survival_integrated import (  # noqa: E402
    CALIBRATION_PROTOCOL,
    CALIBRATION_SCHEMA,
    _validate_image_metric_contract,
    _validate_metrics,
    _validate_subpose_quality_protocol,
    validate_calibration_summary,
    validate_training_artifact,
    validate_variant_semantics,
)


class HistoricalHierarchicalValidatorTests(unittest.TestCase):
    def test_registered_historical_variant_semantics_remain_available(self) -> None:
        result = validate_variant_semantics(
            "R0_free_survival",
            {
                "relationSource": "free",
                "spectralMode": "fourier117",
                "lossVariant": "rvl",
                "effectiveResourceWeight": 0.0,
            },
            {"source": "free", "shuffled": False, "hierarchy": "flat"},
            require_registered=True,
        )
        self.assertEqual(result["relationSource"], "free")
        self.assertEqual(result["spectralMode"], "fourier117")

    def test_historical_calibration_contract_remains_strict(self) -> None:
        summary = {
            "schema": CALIBRATION_SCHEMA,
            "protocol": CALIBRATION_PROTOCOL,
            "selectionRule": "maximize aggregateWeightedRecall under safety floor",
            "status": "safe",
            "thresholdRows": [{
                "threshold": 0.25,
                "aggregateWeightedRecall": 0.995,
                "poseMacroWeightedRecall": 0.994,
                "aggregateWeightedRecallLowerConfidenceBound": 0.993,
                "weightedRecallBootstrapReplicates": 10000,
            }],
            "selected": {
                "threshold": 0.25,
                "aggregateWeightedRecall": 0.995,
                "aggregateWeightedRecallLowerConfidenceBound": 0.993,
            },
            "testRead": False,
        }
        result = validate_calibration_summary(summary, expected_threshold=0.25, allow_unsafe=False)
        self.assertTrue(result["safe"])
        self.assertEqual(result["threshold"], 0.25)

    def test_historical_training_entrypoint_rejects_incomplete_artifact(self) -> None:
        with self.assertRaises(ValueError):
            validate_training_artifact({})

    def test_quality_loss_requires_manifest_and_semantics(self) -> None:
        protocol = {
            "subposeQuality": {
                "required": True,
                "available": True,
                "manifestSha256": "a" * 64,
                "semantics": "visible hit count over represented subposes",
            }
        }
        result = _validate_subpose_quality_protocol(protocol, loss_variant="threshold_aligned_utility")
        self.assertTrue(result["required"])
        self.assertEqual(result["manifestSha256"], "a" * 64)

    def test_rvl_control_may_have_but_must_not_require_sidecar(self) -> None:
        result = _validate_subpose_quality_protocol(
            {"subposeQuality": {"required": False, "available": True}},
            loss_variant="rvl",
        )
        self.assertFalse(result["required"])

    def test_rvl_control_may_omit_legacy_sidecar_block(self) -> None:
        result = _validate_subpose_quality_protocol({}, loss_variant="rvl")
        self.assertFalse(result["required"])
        self.assertFalse(result["available"])
        self.assertIsNone(result["manifestSha256"])

    def test_quality_loss_cannot_omit_sidecar_block(self) -> None:
        with self.assertRaises(ValueError):
            _validate_subpose_quality_protocol({}, loss_variant="quality")

    def test_quality_loss_rejects_missing_sidecar(self) -> None:
        with self.assertRaises(ValueError):
            _validate_subpose_quality_protocol(
                {"subposeQuality": {"required": True, "available": False}},
                loss_variant="quality",
            )

    def test_image_unavailable_is_not_encoded_as_numeric_zero(self) -> None:
        metrics = {
            name: 0.0
            for name in (
                "precision", "recall", "weightedRecall", "f1", "jaccard", "accuracy",
                "balancedAccuracy", "specificity", "usefulCull", "badCull", "avgPredCount",
                "avgCandidateCount", "avgGtCount", "predictedGlbCount", "predictedGlbBytes",
                "glbByteReduction", "requiredGlbCount", "requiredGlbBytes",
            )
        }
        metrics.update({"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0, "weightedTp": 0.0, "weightedGt": 0.0})
        _validate_metrics(metrics, label="fixture")
        _validate_image_metric_contract({
            "metrics": metrics,
            "unavailableMetrics": {"missPixelRate": "not_available"},
        })
        metrics["missPixelRate"] = 0.0
        with self.assertRaises(ValueError):
            _validate_image_metric_contract({
                "metrics": metrics,
                "unavailableMetrics": {"missPixelRate": "not_available"},
            })


if __name__ == "__main__":
    unittest.main()
