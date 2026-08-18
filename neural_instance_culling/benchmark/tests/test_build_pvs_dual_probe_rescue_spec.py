from __future__ import annotations

import unittest

from neural_instance_culling.benchmark.build_pvs_dual_probe_rescue_spec import (
    FEATURE_DIM,
    OUTPUT_SCHEMA,
    build_spec,
)


def _payload(power: float, risk: float) -> dict:
    return {
        "schema": "pvs-difficult-tail-feature-separability-v1",
        "testRead": False,
        "probe": {
            "families": {
                "combined": {
                    "status": "ok",
                    "testRead": False,
                    "fitSplit": "train",
                    "featureCount": FEATURE_DIM,
                    "probeType": "standardized_ridge_linear",
                    "coefficients": [0.0] * (FEATURE_DIM + 1),
                    "standardization": {
                        "mean": [0.0] * FEATURE_DIM,
                        "scale": [1.0] * FEATURE_DIM,
                    },
                    "weighting": {"sampleWeightPower": power},
                    "riskCertificates": {
                        "sourceSplit": "train",
                        "fitRowsOnly": True,
                        "rescue": {
                            str(risk): {
                                "threshold": 0.25,
                                "observedRisk": risk * 0.9,
                                "registeredRiskUpperBound": risk,
                                "riskWithinRegisteredBound": True,
                                "tiePolicy": "inclusive",
                            }
                        },
                    },
                }
            }
        },
    }


class DualProbeRescueSpecTest(unittest.TestCase):
    def test_builds_train_only_runtime_spec(self) -> None:
        result = build_spec(
            _payload(0.5, 0.01),
            _payload(0.0, 0.05),
            alpha=0.5,
            temperature=0.05,
            coverage_weight=2.0,
            primary_rescue_risk=0.01,
            coverage_rescue_risk=0.05,
        )
        self.assertEqual(result["schema"], OUTPUT_SCHEMA)
        self.assertFalse(result["testRead"])
        self.assertFalse(result["selectedFromValidation"])
        self.assertEqual(result["rawFeatureDimension"], FEATURE_DIM)
        self.assertTrue(result["runtime"]["noAdditionalPerInstanceAssets"])
        self.assertEqual(result["primary"]["fitSplit"], "train")
        self.assertEqual(result["primary"]["threshold"], 0.25)
        self.assertTrue(result["primary"]["riskCertificate"]["fitRowsOnly"])
        self.assertEqual(result["alpha"], 0.5)
        self.assertEqual(result["coverageWeight"], 2.0)

    def test_rejects_test_provenance_and_wrong_weight_power(self) -> None:
        primary = _payload(0.5, 0.01)
        primary["testRead"] = True
        with self.assertRaises(ValueError):
            build_spec(
                primary,
                _payload(0.0, 0.05),
                alpha=0.5,
                temperature=0.05,
                coverage_weight=2.0,
                primary_rescue_risk=0.01,
                coverage_rescue_risk=0.05,
            )
        primary = _payload(1.0, 0.01)
        with self.assertRaises(ValueError):
            build_spec(
                primary,
                _payload(0.0, 0.05),
                alpha=0.5,
                temperature=0.05,
                coverage_weight=2.0,
                primary_rescue_risk=0.01,
                coverage_rescue_risk=0.05,
            )

    def test_rejects_non_train_certificate_and_nonpositive_scale(self) -> None:
        primary = _payload(0.5, 0.01)
        primary["probe"]["families"]["combined"]["riskCertificates"]["sourceSplit"] = "calibration"
        with self.assertRaises(ValueError):
            build_spec(
                primary,
                _payload(0.0, 0.05),
                alpha=0.5,
                temperature=0.05,
                coverage_weight=2.0,
                primary_rescue_risk=0.01,
                coverage_rescue_risk=0.05,
            )
        primary = _payload(0.5, 0.01)
        primary["probe"]["families"]["combined"]["standardization"]["scale"][3] = 0.0
        with self.assertRaises(ValueError):
            build_spec(
                primary,
                _payload(0.0, 0.05),
                alpha=0.5,
                temperature=0.05,
                coverage_weight=2.0,
                primary_rescue_risk=0.01,
                coverage_rescue_risk=0.05,
            )


if __name__ == "__main__":
    unittest.main()
