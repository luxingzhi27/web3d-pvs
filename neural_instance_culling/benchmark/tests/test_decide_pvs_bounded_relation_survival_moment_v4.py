from __future__ import annotations

from pathlib import Path
import sys
import unittest

BENCHMARK = Path(__file__).resolve().parents[1]
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))

import decide_pvs_bounded_relation_survival_moment_v4 as route  # noqa: E402


def _safe_member() -> dict:
    return {
        "safeWorkpoint": True,
        "aggregate": {
            "weightedRecall": 0.995,
            "weightedRecallLowerConfidenceBound": 0.991,
        },
        "safety": {},
    }


def _attach_non_worsening_image_effects(metrics: dict) -> None:
    for metric in (
        "meanMissPixelRate", "p95MissPixelRate",
        "meanWrongInstancePixelRate", "p95WrongInstancePixelRate",
        "meanExtraPixelRateOverImage", "p95ExtraPixelRateOverImage",
    ):
        metrics[f"image.{metric}"] = {
            "meanDelta": -0.001,
            "ci95": [-0.002, 0.0],
            "crossesZero": True,
        }


class BoundedRelationSurvivalMomentV3RouteTests(unittest.TestCase):
    def test_registered_policy_self_test(self) -> None:
        result = route.self_test()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["route"], "full_innovation_candidate")

    def test_v4_route_rejects_a_v2_summary_schema(self) -> None:
        with self.assertRaises(ValueError):
            route.make_decision({"schema": "pvs-full-innovation-v2-validation-summary-v1"}, Path("fixture.json"))

    def test_v4_summary_uses_registered_variants_and_paired_bootstrap(self) -> None:
        effects = {}
        for ablation in route.FORMAL_VARIANTS[1:]:
            metrics = {}
            for scope in ("poseMacro", "aggregate"):
                for metric in ("recall", "weightedRecall", "badCull"):
                    metrics[f"{scope}.{metric}"] = {"meanDelta": 0.0, "ci95": [-0.001, 0.001]}
                for metric in ("precision", "balancedAccuracy", "f1", "usefulCull", "predictedGlbBytes", "glbByteReduction", "downloadUtilityRecall"):
                    metrics[f"{scope}.{metric}"] = {"meanDelta": 0.1, "ci95": [0.01, 0.2]}
                metrics[f"{scope}.avgPredCount"] = {"meanDelta": -0.1, "ci95": [-0.2, -0.01]}
            _attach_non_worsening_image_effects(metrics)
            effects[f"full_minus_{ablation}"] = {"metrics": metrics}
        payload = {
            "schema": route.SUMMARY_SCHEMA,
            "split": "validation",
            "testRead": False,
            "variants": list(route.FORMAL_VARIANTS),
            "seeds": list(route.SEEDS),
            "bootstrap": {"replicates": 10000, "paired": True},
            "members": {
                f"{variant}:{seed}": _safe_member()
                for variant in route.FORMAL_VARIANTS
                for seed in route.SEEDS
            },
            "comparisons": effects,
            "unavailableMetrics": {"browserWebGpuLatency": "not attached"},
        }
        result = route.make_decision(payload, Path("v4_fixture.json"))
        self.assertEqual(result["schema"], route.ROUTE_SCHEMA)
        self.assertEqual(result["route"], "full_innovation_candidate")

    def test_pose_recall_bad_cull_and_unsafe_ablation_are_diagnostic_only(self) -> None:
        effects = {}
        for ablation in route.FORMAL_VARIANTS[1:]:
            metrics = {}
            for scope in ("poseMacro", "aggregate"):
                metrics[f"{scope}.recall"] = {"meanDelta": -0.2, "ci95": [-0.3, -0.1]}
                metrics[f"{scope}.badCull"] = {"meanDelta": 0.2, "ci95": [0.1, 0.3]}
                metrics[f"{scope}.weightedRecall"] = {"meanDelta": 0.0, "ci95": [-0.001, 0.001]}
                for metric in ("precision", "balancedAccuracy", "f1", "usefulCull", "glbByteReduction", "downloadUtilityRecall"):
                    metrics[f"{scope}.{metric}"] = {"meanDelta": 0.1, "ci95": [0.01, 0.2]}
                for metric in ("avgPredCount", "predictedGlbBytes"):
                    metrics[f"{scope}.{metric}"] = {"meanDelta": -0.1, "ci95": [-0.2, -0.01]}
            _attach_non_worsening_image_effects(metrics)
            effects[f"full_minus_{ablation}"] = {"metrics": metrics}
        payload = {
            "schema": route.SUMMARY_SCHEMA,
            "split": "validation",
            "testRead": False,
            "variants": list(route.FORMAL_VARIANTS),
            "seeds": list(route.SEEDS),
            "bootstrap": {"replicates": 10000, "paired": True},
            "members": {
                f"{variant}:{seed}": {
                    "safeWorkpoint": variant == "full",
                    "aggregate": {
                        "weightedRecall": 0.995,
                        "weightedRecallLowerConfidenceBound": 0.991,
                    },
                    "safety": {},
                }
                for variant in route.FORMAL_VARIANTS
                for seed in route.SEEDS
            },
            "comparisons": effects,
            "unavailableMetrics": {"browserWebGpuLatency": "not attached"},
        }
        result = route.make_decision(payload, Path("v3_diagnostic_fixture.json"))
        self.assertEqual(result["route"], "full_innovation_candidate")
        self.assertTrue(result["policy"]["poseRecallDiagnosticOnly"])
        self.assertTrue(result["policy"]["badCullConfidenceIntervalDiagnosticOnly"])

    def test_full_weighted_recall_safety_failure_blocks_route(self) -> None:
        fixture = route.self_test()
        self.assertEqual(fixture["route"], "full_innovation_candidate")
        # The full fixture construction lives in self_test; exercising the
        # member gate directly keeps this rejection focused and deterministic.
        summary = {
            "members": {
                f"full:{seed}": {
                    **_safe_member(),
                    "safeWorkpoint": seed != route.SEEDS[0],
                }
                for seed in route.SEEDS
            }
        }
        self.assertFalse(route._member_safety(summary, "full")["allSafe"])

    def test_validation_weighted_recall_or_lcb_failure_blocks_member_safety(self) -> None:
        for failed_field in ("weightedRecall", "weightedRecallLowerConfidenceBound"):
            members = {f"full:{seed}": _safe_member() for seed in route.SEEDS}
            members[f"full:{route.SEEDS[0]}"]["aggregate"][failed_field] = 0.989
            result = route._member_safety({"members": members}, "full")
            self.assertFalse(result["allSafe"], failed_field)
            self.assertTrue(result["allCalibrationSafe"], failed_field)


if __name__ == "__main__":
    unittest.main()
