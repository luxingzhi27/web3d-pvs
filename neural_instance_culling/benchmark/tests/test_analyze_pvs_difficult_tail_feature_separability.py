from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from neural_instance_culling.benchmark.analyze_pvs_difficult_tail_feature_separability import (
    FeatureExamples,
    _boundary_opportunity_values,
    _boundary_tail_residual_values,
    _probe_fit,
    analyze_feature_tables,
    build_train_risk_certificates,
    freeze_tail_cutoffs,
    load_capture,
    region_extrema_features,
)


def _features(values: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    return {
        "center": np.concatenate([values, values + 0.1], axis=1),
        "mean_std": np.concatenate(
            [values, values + 0.2, values + 0.3, values + 0.4], axis=1
        ),
        "boundary": values,
        "region_extrema": np.concatenate([values, values + 0.5, values + 1.0], axis=1),
    }


def _examples(
    split: str,
    scores: list[float],
    targets: list[int],
    values: list[float],
) -> FeatureExamples:
    return FeatureExamples(
        split=split,
        scores=np.asarray(scores, dtype=np.float64),
        targets=np.asarray(targets, dtype=np.float64),
        features=_features(np.asarray(values, dtype=np.float64)),
    )


class DifficultTailFeatureSeparabilityTest(unittest.TestCase):
    def test_train_fit_reports_actual_probe_rows_and_tie_safe_certificates(self) -> None:
        train = FeatureExamples(
            split="train",
            scores=np.asarray([0.01, 0.02, 0.80, 0.90], dtype=np.float64),
            targets=np.asarray([0, 0, 1, 1], dtype=np.float64),
            features={"center": np.asarray([[0.0], [0.5], [1.0], [1.5]])},
            visible_weights=np.asarray([1.0, 4.0, 0.0, 0.0], dtype=np.float64),
        )
        calibration = FeatureExamples(
            split="calibration",
            scores=np.asarray([0.01, 0.02, 0.80, 0.90], dtype=np.float64),
            targets=np.asarray([0, 0, 1, 1], dtype=np.float64),
            features={"center": np.asarray([[0.0], [0.5], [1.0], [1.5]])},
            visible_weights=np.asarray([1.0, 4.0, 0.0, 0.0], dtype=np.float64),
        )
        validation = FeatureExamples(
            split="validation",
            scores=np.asarray([0.01, 0.02, 0.80, 0.90], dtype=np.float64),
            targets=np.asarray([0, 0, 1, 1], dtype=np.float64),
            features={"center": np.asarray([[0.0], [0.5], [1.0], [1.5]])},
            visible_weights=np.asarray([1.0, 4.0, 0.0, 0.0], dtype=np.float64),
        )
        report = _probe_fit(
            train,
            calibration,
            validation,
            "center",
            ridge=1e-3,
            max_samples=-1,
            seed=1,
            sample_weight_power=1.0,
        )
        distribution = report["trainFitProbeScoreDistribution"]
        self.assertEqual(distribution["sourceSplit"], "train")
        self.assertTrue(distribution["fitRowsOnly"])
        self.assertEqual(
            distribution["lowScorePositive"]["count"]
            + distribution["highScoreNegative"]["count"],
            report["fitCount"],
        )
        for group in ("lowScorePositive", "highScoreNegative"):
            self.assertEqual(
                set(group_value for group_value in distribution[group] if group_value.startswith("q")),
                {"q01", "q05", "q25", "q50", "q75", "q95", "q99"},
            )
        self.assertEqual(report["weighting"]["sampleWeightPower"], 1.0)
        fit_weights = np.asarray([0.4, 1.6, 1.0, 1.0], dtype=np.float64)
        fit_values = train.features["center"][:, 0]
        expected_mean = float(np.average(fit_values, weights=fit_weights))
        expected_scale = float(
            np.sqrt(
                np.average(
                    (fit_values - expected_mean) ** 2,
                    weights=fit_weights,
                )
            )
        )
        np.testing.assert_allclose(
            report["standardization"]["mean"], [expected_mean]
        )
        np.testing.assert_allclose(
            report["standardization"]["scale"], [expected_scale]
        )
        self.assertEqual(
            report["weighting"]["standardization"],
            "train_fit_sample_weighted_mean_std",
        )
        self.assertIn("fitEffectiveSampleSize", report["weighting"])
        self.assertIn("fitMaximumWeightShare", report["weighting"])
        self.assertIn("lowScorePositive", report["weighting"]["fitByProbeClass"])
        self.assertIn("highScoreNegative", report["weighting"]["fitByProbeClass"])
        self.assertGreater(report["fit"]["weightedRocAuc"], 0.5)
        certificates = report["riskCertificates"]
        self.assertEqual(certificates["sourceSplit"], "train")
        for kind, risks in (("suppression", ("0.001", "0.005", "0.01")), ("rescue", ("0.01", "0.05"))):
            for risk in risks:
                certificate = certificates[kind][risk]
                self.assertTrue(certificate["riskWithinRegisteredBound"])
                self.assertLessEqual(
                    certificate["observedRisk"],
                    certificate["registeredRiskUpperBound"],
                )

        unweighted = _probe_fit(
            train,
            calibration,
            validation,
            "center",
            ridge=1e-3,
            max_samples=-1,
            seed=1,
        )
        np.testing.assert_allclose(
            unweighted["standardization"]["mean"], [float(np.mean(fit_values))]
        )
        np.testing.assert_allclose(
            unweighted["standardization"]["scale"], [float(np.std(fit_values))]
        )
        self.assertEqual(
            unweighted["weighting"]["standardization"],
            "unweighted_train_fit_mean_std",
        )

    def test_weighted_ridge_can_follow_high_weight_low_score_positive(self) -> None:
        # Most unweighted low-score positives sit at high x, but the one
        # visually important positive at x=0 should determine the weighted
        # tail direction.
        feature_values = np.asarray(
            [
                [6.1227573642, -7.6669950939],
                [1.2542965402, -1.7033088184],
                [-1.3579478763, -0.6467914893],
                [-6.0599583874, -0.6957971329],
                [-2.5956392288, 9.9689985499],
                [0.6773598397, -1.0578923830],
                [-0.8438622545, -2.0041390383],
                [-3.1654516536, -1.1724029317],
                [1.4458361655, -0.7156608197],
            ],
            dtype=np.float64,
        )
        train = FeatureExamples(
            split="train",
            scores=np.asarray([0.01] * 5 + [0.90] * 4),
            targets=np.asarray([0] * 5 + [1] * 4, dtype=np.float64),
            features={"center": feature_values},
            visible_weights=np.asarray([100.0, 1.0, 1.0, 1.0, 1.0] + [0.0] * 4),
        )
        calibration = FeatureExamples(
            split="calibration",
            scores=train.scores.copy(),
            targets=train.targets.copy(),
            features={"center": feature_values.copy()},
            visible_weights=train.visible_weights.copy(),
        )
        validation = FeatureExamples(
            split="validation",
            scores=train.scores.copy(),
            targets=train.targets.copy(),
            features={"center": feature_values.copy()},
            visible_weights=train.visible_weights.copy(),
        )
        unweighted = _probe_fit(
            train, calibration, validation, "center", ridge=1e-3, max_samples=-1, seed=1
        )
        weighted = _probe_fit(
            train,
            calibration,
            validation,
            "center",
            ridge=1e-3,
            max_samples=-1,
            seed=1,
            sample_weight_power=1.0,
        )
        self.assertLess(unweighted["fit"]["weightedRocAuc"], 0.5)
        self.assertGreater(weighted["fit"]["weightedRocAuc"], 0.7)
        self.assertGreater(
            weighted["fit"]["direction"]["weightedHighMinusLow"],
            unweighted["fit"]["direction"]["weightedHighMinusLow"],
        )

    def test_risk_certificates_include_ties_inside_the_observed_risk(self) -> None:
        certificates = build_train_risk_certificates(
            np.asarray([0.1, 0.2, 0.8, 0.8, 0.9], dtype=np.float64),
            np.asarray([0, 0, 0, 1, 1], dtype=np.float64),
            np.asarray([1.0, 1.0, 8.0, 0.0, 0.0], dtype=np.float64),
        )
        suppression = certificates["suppression"]["0.01"]
        rescue = certificates["rescue"]["0.05"]
        self.assertLessEqual(
            suppression["observedRisk"], suppression["registeredRiskUpperBound"]
        )
        self.assertLessEqual(
            rescue["observedRisk"], rescue["registeredRiskUpperBound"]
        )
        self.assertEqual(suppression["tiePolicy"].split(";")[0], "inclusive observed threshold")
        self.assertEqual(rescue["tiePolicy"].split(";")[0], "inclusive observed threshold")

    def test_optional_boundary_heads_are_reconstructed_from_checkpoint_config(self) -> None:
        config = {
            "boundaryOpportunity": {
                "enabled": True,
                "hiddenDim": 18,
                "projectionDim": 12,
                "initialLogit": -4.0,
                "maximumLogitUplift": 3.5,
            },
            "boundaryTailResidual": {
                "enabled": True,
                "hiddenDim": 32,
                "projectionDim": 24,
                "maximumAbsoluteResidual": 1.5,
                "centering": "none",
                "shortcut": "region_linear",
                "fusionMode": "affine_region",
                "outputInitializationStd": 0.002,
            },
        }
        self.assertEqual(_boundary_opportunity_values(config), (18, 12, -4.0, 3.5))
        self.assertEqual(
            _boundary_tail_residual_values(config),
            (32, 24, 1.5, "none", "region_linear", "affine_region", 0.002),
        )

    def test_disabled_boundary_heads_use_registered_constructor_defaults(self) -> None:
        self.assertEqual(_boundary_opportunity_values({}), (0, 24, -6.0, 6.0))
        self.assertEqual(
            _boundary_tail_residual_values({}),
            (0, 24, 1.0, "pose_mean", "none", "product", 0.0),
        )

    def test_tail_cutoffs_are_frozen_from_calibration_only(self) -> None:
        cutoffs = freeze_tail_cutoffs(
            np.asarray([0.01, 0.20, 0.80, 0.90]),
            np.asarray([1, 1, 0, 0]),
            positive_quantile=0.5,
            negative_quantile=0.5,
        )
        self.assertEqual(cutoffs["sourceSplit"], "calibration")
        self.assertFalse(cutoffs["selectedFromValidation"])
        self.assertAlmostEqual(cutoffs["lowPositiveScoreCutoff"], 0.105)
        self.assertAlmostEqual(cutoffs["highNegativeScoreCutoff"], 0.85)

    def test_weighted_positive_tail_uses_visible_mass_not_instance_count(self) -> None:
        cutoffs = freeze_tail_cutoffs(
            np.asarray([0.01, 0.20, 0.80, 0.90, 0.95]),
            np.asarray([1, 1, 1, 0, 0]),
            visible_weights=np.asarray([0.01, 0.01, 0.98, 0.0, 0.0]),
            positive_tail_mode="weighted_mass",
            positive_quantile=0.5,
            negative_quantile=0.5,
        )
        self.assertEqual(cutoffs["positiveTailMode"], "weighted_mass")
        self.assertAlmostEqual(cutoffs["lowPositiveScoreCutoff"], 0.80)

    def test_region_extrema_are_the_unit_disk_coordinate_bounds(self) -> None:
        center = np.asarray([[0.2, -0.4]], dtype=np.float64)
        axes = np.asarray([[[0.3, 0.4], [0.0, 0.5]]], dtype=np.float64)
        result = region_extrema_features(center, axes)
        np.testing.assert_allclose(
            result,
            np.asarray([[-0.3, -0.9, 0.7, 0.1, 1.0, 1.0]]),
        )

    def test_report_contains_group_quantiles_effect_sizes_and_validation_probe(self) -> None:
        calibration = _examples(
            "calibration",
            [0.01, 0.02, 0.40, 0.95, 0.96, 0.97],
            [1, 1, 0, 0, 0, 0],
            [0.0, 0.1, 0.2, 1.0, 1.1, 1.2],
        )
        validation = _examples(
            "validation",
            [0.015, 0.03, 0.45, 0.94, 0.965, 0.98],
            [1, 1, 0, 0, 0, 0],
            [0.05, 0.12, 0.25, 1.05, 1.15, 1.25],
        )
        report = analyze_feature_tables(
            calibration,
            validation,
            positive_tail_quantile=0.5,
            negative_tail_quantile=0.5,
            ridge=1e-2,
        )
        self.assertFalse(report["testRead"])
        self.assertEqual(report["groups"]["calibration"]["lowScorePositive"], 1)
        self.assertEqual(report["groups"]["validation"]["highScoreNegative"], 2)
        center = report["features"]["center"]
        self.assertIn("q50", center["calibration"]["lowScorePositive"]["quantiles"])
        self.assertEqual(
            len(center["validation"]["standardizedMeanDifference"]),
            2,
        )
        probe = report["probe"]["families"]["boundary"]["validation"]
        self.assertIsNotNone(probe["rocAuc"])
        self.assertIsNotNone(probe["averagePrecision"])
        self.assertEqual(
            len(report["probe"]["families"]["boundary"]["coefficients"]),
            report["probe"]["families"]["boundary"]["featureCount"] + 1,
        )
        self.assertEqual(report["probe"]["thresholdSource"], "calibration")
        self.assertFalse(report["probe"]["selectedFromValidation"])

    def test_train_probe_uses_train_features_but_calibration_threshold(self) -> None:
        calibration = _examples(
            "calibration",
            [0.01, 0.30, 0.70, 0.99],
            [1, 0, 1, 0],
            [0.0, 0.4, 0.2, 1.0],
        )
        validation = _examples(
            "validation",
            [0.02, 0.35, 0.75, 0.98],
            [1, 0, 1, 0],
            [0.1, 0.45, 0.3, 1.1],
        )
        train = _examples(
            "train",
            [0.015, 0.4, 0.72, 0.97],
            [1, 0, 1, 0],
            [0.05, 0.5, 0.25, 1.05],
        )
        report = analyze_feature_tables(
            calibration,
            validation,
            train=train,
            positive_tail_quantile=0.5,
            negative_tail_quantile=0.5,
            probe_fit_split="train",
            probe_fit_tail_source="train",
        )
        self.assertEqual(report["probe"]["fitSplit"], "train")
        self.assertEqual(report["probe"]["families"]["center"]["fitSplit"], "train")
        self.assertEqual(
            report["tailDefinition"]["fitCutoffs"]["sourceSplit"],
            "train",
        )
        threshold = report["probe"]["families"]["center"]["frozenThreshold"]
        self.assertEqual(threshold["sourceSplit"], "calibration")
        self.assertFalse(threshold["selectedFromValidation"])

    def test_capture_rejects_test_even_when_test_read_flag_is_false(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.json"
            path.write_text(
                json.dumps(
                    {
                        "split": "test",
                        "testRead": False,
                        "perPose": [{"poseIndex": 0}],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_capture(path, "test")


if __name__ == "__main__":
    unittest.main()
