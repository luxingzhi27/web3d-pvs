from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from neural_instance_culling.benchmark.scan_pvs_train_owned_tail_residual import (
    ALLOWED_SPLITS,
    PROBE_LABEL_SEMANTICS,
    PROBE_SCHEMA,
    SCHEMA,
    SUPPORTED_RESIDUAL_MODES,
    SUPPORTED_FAMILIES,
    CandidateSplit,
    ProbeSpec,
    apply_bounded_centered_residual,
    build_probe_feature_matrix,
    certificate_threshold_from_probe,
    frontier_center_from_probe_payload,
    freeze_calibration_workpoint,
    load_probe_spec,
    load_score_rows,
    probe_linear_scores,
    residual_diagnostics,
    score_probe_families_from_aux,
    scan_members,
    summarize_threshold,
)
import neural_instance_culling.benchmark.scan_pvs_train_owned_tail_residual as scan_module
from neural_instance_culling.benchmark.analyze_pvs_difficult_tail_feature_separability import (
    ScoreRow,
)


def _row(pose: int, scores: list[float], targets: list[int], weights: list[float]) -> ScoreRow:
    return ScoreRow(
        pose_index=pose,
        candidate_ids=np.arange(len(scores), dtype=np.int64),
        scores=np.asarray(scores, dtype=np.float64),
        targets=np.asarray(targets, dtype=np.float64),
        visible_weights=np.asarray(weights, dtype=np.float64),
    )


def _probe_payload(feature_count: int = 27) -> dict:
    return {
        "schema": PROBE_SCHEMA,
        "testRead": False,
        "splitsRead": list(ALLOWED_SPLITS),
        "tailDefinition": {
            "probeLabel": PROBE_LABEL_SEMANTICS,
            "fitCutoffs": {
                "sourceSplit": "train",
                "lowPositiveScoreCutoff": 0.1,
                "highNegativeScoreCutoff": 0.2,
            },
        },
        "probe": {
            "fitSplit": "train",
            "thresholdSource": "calibration",
            "selectedFromValidation": False,
            "combinedFeatureNames": [f"feature_{i}" for i in range(108)],
                "families": {
                    "region_extrema": {
                    "status": "ok",
                    "fitSplit": "train",
                    "featureCount": feature_count,
                    "coefficients": [0.25] + [0.1] * feature_count,
                    "ridge": 0.001,
                        "standardization": {
                            "mean": [0.0] * feature_count,
                            "scale": [1.0] * feature_count,
                        },
                        "riskCertificates": {
                            "sourceSplit": "train",
                            "fitRowsOnly": True,
                            "suppression": {
                                "0.001": {
                                    "threshold": 2.0,
                                    "observedRisk": 0.0,
                                    "registeredRiskUpperBound": 0.001,
                                    "riskWithinRegisteredBound": True,
                                },
                                "0.005": {
                                    "threshold": 1.5,
                                    "observedRisk": 0.0,
                                    "registeredRiskUpperBound": 0.005,
                                    "riskWithinRegisteredBound": True,
                                },
                                "0.01": {
                                    "threshold": 1.0,
                                    "observedRisk": 0.0,
                                    "registeredRiskUpperBound": 0.01,
                                    "riskWithinRegisteredBound": True,
                                },
                            },
                            "rescue": {
                                "0.01": {
                                    "threshold": -1.0,
                                    "observedRisk": 0.0,
                                    "registeredRiskUpperBound": 0.01,
                                    "riskWithinRegisteredBound": True,
                                },
                                "0.05": {
                                    "threshold": 0.0,
                                    "observedRisk": 0.0,
                                    "registeredRiskUpperBound": 0.05,
                                    "riskWithinRegisteredBound": True,
                                },
                            },
                        },
                    },
                },
        },
    }


class TrainOwnedTailResidualTest(unittest.TestCase):
    def test_schema_and_supported_family_symbols_are_explicit(self) -> None:
        self.assertEqual(SCHEMA, "pvs-train-owned-tail-residual-posterior-scan-v1")
        self.assertEqual(ALLOWED_SPLITS, ("train", "calibration", "validation"))
        self.assertEqual(SUPPORTED_FAMILIES, ("region_extrema", "combined"))
        self.assertEqual(
            SUPPORTED_RESIDUAL_MODES,
            ("symmetric", "rescue_only", "suppress_only", "dual", "dual_rescue"),
        )

    def test_dual_rescue_cli_rejects_missing_coverage_probe(self) -> None:
        from neural_instance_culling.benchmark.scan_pvs_train_owned_tail_residual import main

        with self.assertRaisesRegex(ValueError, "coverage-probe"):
            main(
                [
                    "--train-capture", "unused-train.json",
                    "--calibration-capture", "unused-calibration.json",
                    "--validation-capture", "unused-validation.json",
                    "--probe", "unused-probe.json",
                    "--checkpoint", "unused-checkpoint.pt",
                    "--dataset-dir", "unused-dataset",
                    "--runtime-meta", "unused-runtime-meta.json",
                    "--initial-geo-features", "unused-geo.bin",
                    "--output", "unused-output.json",
                    "--residual-modes", "dual_rescue",
                ]
            )

    def test_same_probe_family_reconstructs_feature_matrix_once(self) -> None:
        probe_a = ProbeSpec(
            family="region_extrema",
            feature_count=27,
            coefficients=np.asarray([0.0] + [0.1] * 27),
            mean=np.zeros(27),
            scale=np.ones(27),
            fit_split="train",
            ridge=0.001,
        )
        probe_b = ProbeSpec(
            family="region_extrema",
            feature_count=27,
            coefficients=np.asarray([1.0] + [0.2] * 27),
            mean=np.zeros(27),
            scale=np.ones(27),
            fit_split="train",
            ridge=0.002,
        )
        aux = {
            "center_view": np.zeros((2, 9), dtype=np.float64),
            "spectral_features": np.zeros((2, 64), dtype=np.float64),
            "boundary_spectral_summary": np.zeros((2, 8), dtype=np.float64),
            "disk_axes": np.ones((2, 9, 2), dtype=np.float64),
        }
        with patch.object(
            scan_module,
            "build_probe_feature_matrix",
            wraps=build_probe_feature_matrix,
        ) as builder:
            scores = score_probe_families_from_aux(
                aux,
                {"primary": probe_a, "coverage": probe_b},
            )
        self.assertEqual(builder.call_count, 1)
        self.assertEqual(set(scores), {"primary", "coverage"})
        self.assertEqual(scores["primary"].shape, (2,))
        self.assertEqual(scores["coverage"].shape, (2,))

    def test_train_probe_parameters_and_frontier_center_use_correct_spaces(self) -> None:
        payload = _probe_payload()
        expected = 0.5 * (
            np.log(0.1 / 0.9) + np.log(0.2 / 0.8)
        )
        self.assertAlmostEqual(frontier_center_from_probe_payload(payload), expected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            probe, _ = load_probe_spec(path, "region_extrema")
        self.assertEqual(probe.fit_split, "train")
        self.assertEqual(probe.feature_count, 27)
        self.assertAlmostEqual(probe.coefficients[0], 0.25)
        self.assertAlmostEqual(probe.frontier_center_logit or 0.0, expected)
        features = np.ones((2, 27), dtype=np.float64)
        np.testing.assert_allclose(probe_linear_scores(features, probe), 0.25 + 2.7)

    def test_hinge_probe_uses_train_fixed_piecewise_linear_basis(self) -> None:
        probe = ProbeSpec(
            family="region_extrema",
            feature_count=2,
            coefficients=np.asarray([0.5, 1.0, 2.0, 3.0, 4.0]),
            mean=np.zeros(2),
            scale=np.ones(2),
            fit_split="train",
            ridge=0.001,
            probe_type="standardized_ridge_hinge",
            hinge_knots=np.asarray([0.0]),
        )
        features = np.asarray([[-1.0, 2.0], [3.0, -2.0]], dtype=np.float64)
        expected = np.asarray([
            0.5 - 1.0 + 4.0 + 0.0 + 8.0,
            0.5 + 3.0 - 4.0 + 9.0 + 0.0,
        ])
        np.testing.assert_allclose(probe_linear_scores(features, probe), expected)

    def test_shallow_mlp_probe_uses_exported_relu_parameters(self) -> None:
        probe = ProbeSpec(
            family="region_extrema",
            feature_count=2,
            coefficients=np.zeros(0),
            mean=np.zeros(2),
            scale=np.ones(2),
            fit_split="train",
            ridge=0.001,
            probe_type="standardized_shallow_mlp",
            hidden_weight=np.asarray([[1.0, -1.0], [-1.0, 1.0]]),
            hidden_bias=np.asarray([0.0, 0.5]),
            output_weight=np.asarray([2.0, -1.0]),
            output_bias=0.25,
            activation="relu",
        )
        features = np.asarray([[2.0, 1.0], [0.0, 2.0]], dtype=np.float64)
        expected = np.asarray([2.25, -2.25])
        np.testing.assert_allclose(probe_linear_scores(features, probe), expected)

    def test_feature_families_have_expected_candidate_alignment(self) -> None:
        batch = 3
        aux = {
            "center_view": np.zeros((batch, 9), dtype=np.float64),
            "spectral_features": np.zeros((batch, 64), dtype=np.float64),
            "boundary_spectral_summary": np.zeros((batch, 8), dtype=np.float64),
            "disk_axes": np.ones((batch, 9, 2), dtype=np.float64),
        }
        region = build_probe_feature_matrix(aux, "region_extrema")
        combined = build_probe_feature_matrix(aux, "combined")
        self.assertEqual(region.shape, (batch, 27))
        self.assertEqual(combined.shape, (batch, 108))

    def test_residual_is_bounded_centered_and_reverse_sign_is_a_real_control(self) -> None:
        base = np.full((4,), 0.5, dtype=np.float64)
        probe = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
        poses = np.asarray([10, 10, 20, 20], dtype=np.int64)
        corrected, residual = apply_bounded_centered_residual(
            base,
            probe,
            poses,
            alpha=0.5,
            temperature=0.1,
            residual_sign=1.0,
        )
        np.testing.assert_allclose(residual, [0.5, -0.5, 0.5, -0.5], atol=1e-4)
        self.assertGreater(corrected[0], 0.5)
        self.assertLess(corrected[1], 0.5)
        self.assertAlmostEqual(float(residual[:2].mean()), 0.0)
        self.assertAlmostEqual(float(residual[2:].mean()), 0.0)
        _reversed, reverse_residual = apply_bounded_centered_residual(
            base,
            probe,
            poses,
            alpha=0.5,
            temperature=0.1,
            residual_sign=-1.0,
        )
        np.testing.assert_allclose(reverse_residual, -residual, atol=1e-6)

    def test_one_sided_modes_use_certificate_direction_without_centering(self) -> None:
        base = np.full((4,), 0.5, dtype=np.float64)
        probe = np.asarray([-1.0, 0.0, 1.0, 2.0], dtype=np.float64)
        poses = np.asarray([10, 10, 20, 20], dtype=np.int64)
        rescued, rescue_residual = apply_bounded_centered_residual(
            base,
            probe,
            poses,
            alpha=0.5,
            temperature=0.1,
            mode="rescue_only",
            rescue_threshold=0.0,
        )
        suppressed, suppress_residual = apply_bounded_centered_residual(
            base,
            probe,
            poses,
            alpha=0.5,
            temperature=0.1,
            mode="suppress_only",
            suppression_threshold=1.0,
        )
        _dual, dual_residual = apply_bounded_centered_residual(
            base,
            probe,
            poses,
            alpha=0.5,
            temperature=0.1,
            mode="dual",
            suppression_threshold=1.0,
            rescue_threshold=0.0,
        )
        self.assertTrue(np.all(rescue_residual >= 0.0))
        self.assertTrue(np.all(suppress_residual <= 0.0))
        self.assertGreater(rescue_residual[0], 0.0)
        self.assertAlmostEqual(float(rescue_residual[2]), 0.0)
        self.assertLess(suppress_residual[3], 0.0)
        self.assertAlmostEqual(float(suppress_residual[0]), 0.0)
        np.testing.assert_allclose(dual_residual, rescue_residual + suppress_residual)
        self.assertGreater(rescued[0], base[0])
        self.assertLess(suppressed[3], base[3])

    def test_dual_rescue_is_non_negative_and_coverage_branch_is_active(self) -> None:
        base = np.full((3,), 0.5, dtype=np.float64)
        primary = np.asarray([1.0, 1.0, -1.0], dtype=np.float64)
        coverage = np.asarray([-1.0, 1.0, 1.0], dtype=np.float64)
        poses = np.asarray([0, 0, 0], dtype=np.int64)
        _corrected_low, residual_low = apply_bounded_centered_residual(
            base,
            primary,
            poses,
            alpha=0.5,
            temperature=0.1,
            mode="dual_rescue",
            rescue_threshold=0.0,
            coverage_probe_scores=coverage,
            coverage_threshold=0.0,
            coverage_weight=0.25,
        )
        _corrected_high, residual_high = apply_bounded_centered_residual(
            base,
            primary,
            poses,
            alpha=0.5,
            temperature=0.1,
            mode="dual_rescue",
            rescue_threshold=0.0,
            coverage_probe_scores=coverage,
            coverage_threshold=0.0,
            coverage_weight=1.0,
        )
        self.assertTrue(np.all(residual_low >= 0.0))
        self.assertTrue(np.all(residual_high >= 0.0))
        # Candidate 0 has no primary rescue gate but is rescued by coverage.
        self.assertAlmostEqual(float(residual_low[0]), 0.125, places=4)
        self.assertGreater(float(residual_high[0]), float(residual_low[0]))
        # Candidate 2 is rescued by the primary probe, independently of coverage.
        self.assertGreater(float(residual_low[2]), 0.0)

    def test_dual_rescue_scan_reports_two_train_certificate_diagnostics(self) -> None:
        calibration_rows = [
            _row(0, [0.90, 0.10], [1, 0], [1.0, 0.0]),
            _row(1, [0.80, 0.10], [1, 0], [1.0, 0.0]),
        ]
        validation_rows = [
            _row(2, [0.90, 0.10], [1, 0], [1.0, 0.0]),
            _row(3, [0.80, 0.20], [1, 0], [1.0, 0.0]),
        ]
        calibration = CandidateSplit(
            "calibration",
            calibration_rows,
            [np.asarray([1.0, 1.0]), np.asarray([1.0, 1.0])],
            [np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0])],
        )
        validation = CandidateSplit(
            "validation",
            validation_rows,
            [np.asarray([1.0, 1.0]), np.asarray([1.0, 1.0])],
            [np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0])],
        )
        train_cutoffs = {
            "sourceSplit": "train",
            "lowPositiveScoreCutoff": 0.2,
            "highNegativeScoreCutoff": 0.8,
        }
        calibration_cutoffs = {
            "sourceSplit": "calibration",
            "lowPositiveScoreCutoff": 0.2,
            "highNegativeScoreCutoff": 0.8,
        }
        members = scan_members(
            calibration,
            validation,
            alphas=[0.25],
            temperatures=[0.1],
            gate_temperatures=[None],
            residual_modes=["dual_rescue"],
            rescue_threshold=0.5,
            coverage_rescue_threshold=0.5,
            rescue_risk=0.05,
            coverage_rescue_risk=0.05,
            coverage_rescue_weights=[1.0],
            train_fit_cutoffs=train_cutoffs,
            calibration_cutoffs=calibration_cutoffs,
            coverage_train_fit_cutoffs=train_cutoffs,
            coverage_calibration_cutoffs=calibration_cutoffs,
            bootstrap_replicates=20,
            seed=17,
        )
        self.assertEqual(len(members), 1)
        member = members[0]
        self.assertEqual(member["mode"], "dual_rescue")
        self.assertEqual(member["coverageWeight"], 1.0)
        self.assertEqual(
            member["certificate"]["primary"]["thresholdSource"],
            "train_probe_risk_certificate",
        )
        self.assertEqual(
            member["certificate"]["coverage"]["thresholdSource"],
            "coverage_probe_train_risk_certificate",
        )
        self.assertIn("primaryProbe", member["residualDiagnostics"]["calibration"])
        self.assertIn("coverageProbe", member["residualDiagnostics"]["validation"])

    def test_non_train_owned_coverage_certificate_is_rejected(self) -> None:
        payload = _probe_payload()
        payload["probe"]["families"]["region_extrema"]["riskCertificates"]["sourceSplit"] = (
            "calibration"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "coverage-probe.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "train-only"):
                load_probe_spec(path, "region_extrema")

    def test_scan_one_sided_mode_keeps_certificate_threshold_and_validation_lcb(self) -> None:
        calibration_rows = [
            _row(0, [0.90, 0.10], [1, 0], [1.0, 0.0]),
            _row(1, [0.80, 0.10], [1, 0], [1.0, 0.0]),
        ]
        validation_rows = [
            _row(2, [0.90, 0.10], [1, 0], [1.0, 0.0]),
            _row(3, [0.80, 0.20], [1, 0], [1.0, 0.0]),
        ]
        calibration = CandidateSplit(
            "calibration",
            calibration_rows,
            [np.asarray([0.0, 0.0]), np.asarray([0.0, 0.0])],
        )
        validation = CandidateSplit(
            "validation",
            validation_rows,
            [np.asarray([0.0, 0.0]), np.asarray([0.0, 0.0])],
        )
        members = scan_members(
            calibration,
            validation,
            alphas=[0.25],
            temperatures=[0.1],
            gate_temperatures=[None],
            residual_modes=["rescue_only"],
            rescue_threshold=0.5,
            rescue_risk=0.05,
            train_fit_cutoffs={
                "sourceSplit": "train",
                "lowPositiveScoreCutoff": 0.2,
                "highNegativeScoreCutoff": 0.8,
            },
            calibration_cutoffs={
                "sourceSplit": "calibration",
                "lowPositiveScoreCutoff": 0.2,
                "highNegativeScoreCutoff": 0.8,
            },
            bootstrap_replicates=20,
            seed=17,
        )
        self.assertEqual(len(members), 1)
        member = members[0]
        self.assertEqual(member["mode"], "rescue_only")
        self.assertEqual(member["certificate"]["thresholdSource"], "train_probe_risk_certificate")
        self.assertEqual(member["certificate"]["rescueThreshold"], 0.5)
        self.assertIsNotNone(member["validation"])
        self.assertIn("weightedRecallLowerConfidenceBound", member["validation"])
        self.assertIn("trainFitCutoffs", member["residualDiagnostics"]["calibration"])
        self.assertIn("calibrationCutoffs", member["residualDiagnostics"]["validation"])

    def test_certificate_threshold_is_train_owned_and_residual_diagnostics_report_both_cutoffs(self) -> None:
        payload = _probe_payload()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            probe, _ = load_probe_spec(path, "region_extrema")
        self.assertEqual(
            certificate_threshold_from_probe(probe, kind="suppression", risk=0.01),
            1.0,
        )
        self.assertEqual(
            certificate_threshold_from_probe(probe, kind="rescue", risk=0.05),
            0.0,
        )
        rows = [_row(0, [0.1, 0.9, 0.5], [1, 0, 1], [1.0, 0.0, 1.0])]
        # Probe values deliberately disagree with the captured visibility
        # scores.  Tail membership must use the latter, and the two
        # class-conditional score cutoffs are allowed to cross.
        split = CandidateSplit(
            "calibration", rows, [np.asarray([1.0, 0.0, 1.0], dtype=np.float64)]
        )
        diagnostics = residual_diagnostics(
            rows,
            split.probe_scores or [],
            [np.asarray([0.2, -0.3, 0.4])],
            [np.asarray([0.1, -0.2, 0.3])],
            {"sourceSplit": "train", "lowPositiveScoreCutoff": 0.8, "highNegativeScoreCutoff": 0.2},
            cutoff_source="trainFitCutoffs",
        )
        self.assertEqual(diagnostics["cutoffSource"], "trainFitCutoffs")
        self.assertEqual(diagnostics["lowScorePositive"]["count"], 2)
        self.assertEqual(diagnostics["highScoreNegative"]["count"], 1)
        self.assertAlmostEqual(
            diagnostics["lowScorePositive"]["actualMean"], 0.2
        )
        self.assertAlmostEqual(
            diagnostics["highScoreNegative"]["actualMean"], -0.2
        )

    def test_frontier_gate_is_train_only_and_uses_base_logit(self) -> None:
        base = np.asarray([0.1, 0.2, 0.1, 0.2], dtype=np.float64)
        probe = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
        poses = np.asarray([0, 0, 0, 0], dtype=np.int64)
        _corrected, residual = apply_bounded_centered_residual(
            base,
            probe,
            poses,
            alpha=0.5,
            temperature=0.1,
            gate_temperature=0.5,
            frontier_center_logit=frontier_center_from_probe_payload(_probe_payload()),
        )
        self.assertTrue(np.isfinite(residual).all())
        self.assertAlmostEqual(float(residual.mean()), 0.0)
        with self.assertRaises(ValueError):
            apply_bounded_centered_residual(
                base,
                probe,
                poses,
                alpha=0.5,
                temperature=0.1,
                gate_temperature=0.5,
                frontier_center_logit=None,
            )

    def test_test_split_is_rejected_before_dataset_access(self) -> None:
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
                load_score_rows(path, "test", object())
            probe = _probe_payload()
            probe["splitsRead"] = ["train", "calibration", "validation", "test"]
            probe_path = Path(directory) / "probe.json"
            probe_path.write_text(json.dumps(probe), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_probe_spec(probe_path, "region_extrema")

    def test_each_member_freezes_threshold_on_calibration_before_validation(self) -> None:
        calibration_rows = [
            _row(0, [0.90, 0.10], [1, 0], [1.0, 0.0]),
            _row(1, [0.80, 0.10], [1, 0], [1.0, 0.0]),
        ]
        validation_rows = [
            _row(2, [0.90, 0.10], [1, 0], [1.0, 0.0]),
            _row(3, [0.80, 0.90], [1, 0], [1.0, 0.0]),
        ]
        calibration = CandidateSplit(
            "calibration", calibration_rows, [np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0])]
        )
        validation = CandidateSplit(
            "validation", validation_rows, [np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0])]
        )
        members = scan_members(
            calibration,
            validation,
            alphas=[0.0],
            temperatures=[0.1],
            gate_temperatures=[None],
            residual_signs=[1.0, -1.0],
            bootstrap_replicates=100,
            seed=7,
        )
        self.assertEqual(len(members), 2)
        self.assertEqual({member["residualSign"] for member in members}, {1.0, -1.0})
        for member in members:
            calibration_record = member["calibration"]
            self.assertEqual(calibration_record["selectionSplit"], "calibration")
            self.assertFalse(calibration_record["selectedFromValidation"])
            self.assertEqual(calibration_record["status"], "safe")
            self.assertIsNotNone(member["validation"])
            self.assertEqual(member["validation"]["threshold"], calibration_record["selected"]["threshold"])
            self.assertIn("weightedRecallLowerConfidenceBound", member["validation"])
            self.assertNotIn("_weightedTpByPose", member["validation"])
            self.assertNotIn("_weightedGtByPose", member["validation"])
            self.assertEqual(
                [row["poseIndex"] for row in member["validation"]["perPose"]],
                [2, 3],
            )
        self.assertEqual(
            members[0]["calibration"]["selected"],
            members[1]["calibration"]["selected"],
            "alpha=0 policies must not depend on member order or residual sign",
        )

    def test_threshold_summary_persists_candidate_aligned_pose_statistics(self) -> None:
        rows = [
            _row(11, [0.8, 0.2], [1, 0], [3.0, 0.0]),
            _row(17, [0.4, 0.6], [1, 0], [2.0, 0.0]),
        ]
        summary = summarize_threshold(rows, [row.scores for row in rows], 0.5)
        self.assertEqual([row["poseIndex"] for row in summary["perPose"]], [11, 17])
        self.assertEqual(summary["perPose"][0]["tp"], 1)
        self.assertEqual(summary["perPose"][1]["fp"], 1)
        self.assertEqual(summary["aggregate"]["candidateCount"], 4)

    def test_direct_calibration_freeze_has_no_validation_argument(self) -> None:
        rows = [_row(0, [0.9, 0.1], [1, 0], [1.0, 0.0])]
        scores = [np.asarray([0.9, 0.1], dtype=np.float64)]
        workpoint = freeze_calibration_workpoint(rows, scores, bootstrap_replicates=20, seed=1)
        self.assertEqual(workpoint["selectionSplit"], "calibration")
        self.assertFalse(workpoint["selectedFromValidation"])
        self.assertEqual(workpoint["testRead"], False)


if __name__ == "__main__":
    unittest.main()
