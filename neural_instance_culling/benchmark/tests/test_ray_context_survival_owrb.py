from __future__ import annotations

import sys
import unittest
from pathlib import Path

BENCHMARK = Path(__file__).resolve().parents[1]
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))
from decide_ray_context_survival_owrb_route import make_decision, self_test as route_self_test
from evaluate_ray_context_survival_owrb import _summarize
from run_ray_context_survival_owrb_matrix import FORMAL_VARIANTS, _comparisons
from summarize_ray_context_survival_owrb import (
    _bootstrap_effect,
    _factor_contrasts,
    _validation_member_safety,
    self_test as summary_self_test,
)


class RayContextSurvivalOWRBBenchmarkTest(unittest.TestCase):
    def test_summary_self_test(self) -> None:
        self.assertEqual(summary_self_test()["status"], "ok")

    def test_route_self_test(self) -> None:
        self.assertEqual(route_self_test()["status"], "ok")

    def test_pose_macro_exposes_average_confusion_counts(self) -> None:
        payload = _summarize([{
            "poseIndex": 0, "tp": 2, "fp": 3, "fn": 4, "tn": 5,
            "weightedTp": 2.0, "weightedGt": 2.0,
            "downloadUtilityTp": 2.0,
            "downloadUtilityRecall": 1.0,
            "glbBytesAtAchievedVisualUtility": 10.0,
            "precision": 0.4, "recall": 1.0 / 3.0,
            "weightedRecall": 1.0, "f1": 0.5, "jaccard": 2.0 / 9.0,
            "accuracy": 0.7, "specificity": 5.0 / 8.0,
            "balancedAccuracy": 0.4791666667, "usefulCull": 5.0 / 14.0,
            "badCull": 4.0 / 14.0, "predCount": 5,
            "candidateCount": 14, "gtCount": 6,
            "predOverCandidate": 5.0 / 14.0, "predOverGt": 5.0 / 6.0,
            "candidateGlbCount": 1, "predictedGlbCount": 1,
            "candidateGlbBytes": 10.0, "predictedGlbBytes": 10.0,
            "glbCountReduction": 0.0, "glbByteReduction": 0.0,
            "forwardLatencyMs": 1.0,
        }])
        for key, value in {"tp": 2.0, "fp": 3.0, "fn": 4.0, "tn": 5.0}.items():
            self.assertEqual(payload["poseMacro"][key], value)

    def test_formal_matrix_is_complete_two_by_two_by_two(self) -> None:
        self.assertEqual(len(FORMAL_VARIANTS), 8)
        factor_values = set()
        for name, spec in FORMAL_VARIANTS.items():
            self.assertRegex(name, r"^context_(off|on)_survival_(off|on)_(rvl|safety)$")
            factor_values.add((
                spec["contextMode"],
                bool(spec.get("disableSurvival", False)),
                spec["lossMode"],
            ))
        self.assertEqual(len(factor_values), 8)
        self.assertEqual(len(_comparisons(FORMAL_VARIANTS)), 12)

    def test_route_reads_registered_factor_effects(self) -> None:
        metrics = {}
        for metric in (
            "pose_weighted_recall", "aggregate_weighted_recall", "pose_recall", "pose_bad_cull",
            "pose_precision", "aggregate_precision", "pose_accuracy", "aggregate_accuracy",
            "pose_balanced_accuracy", "aggregate_balanced_accuracy", "pose_specificity", "aggregate_specificity",
            "pose_f1", "aggregate_f1", "pose_useful_cull", "aggregate_useful_cull",
            "pose_avg_pred_count", "aggregate_avg_pred_count", "pose_predicted_glb_bytes", "aggregate_predicted_glb_bytes",
            "pose_predicted_glb_count", "aggregate_predicted_glb_count",
            "pose_glb_bytes_at_achieved_visual_utility", "aggregate_glb_bytes_at_achieved_visual_utility",
            "pose_download_utility_recall", "aggregate_download_utility_recall",
        ):
            metrics[metric] = {"meanDelta": 0.01, "ci95": [0.001, 0.02], "crossesZero": False}
        summary = {
            "testRead": False,
            "comparisons": {},
            "factorEffects": {
                "context_main_effect": {
                    "metrics": metrics,
                    "safeWorkpoints": {"bothAllSafe": True},
                },
                "survival_main_effect": {
                    "metrics": metrics,
                    "safeWorkpoints": {"bothAllSafe": True},
                },
                "loss_main_effect": {
                    "metrics": metrics,
                    "safeWorkpoints": {"bothAllSafe": True},
                },
            },
        }
        result = make_decision(summary)
        self.assertEqual(result["route"], "retain_ray_context_survival_owrb_candidate")
        self.assertEqual(set(result["retainedFactors"]), {"context", "survivalField", "loss"})

    def test_factor_effects_keep_safety_qualification_at_top_level(self) -> None:
        """The summary/route contract must not hide factor safety metadata."""
        members = {}
        for context in ("off", "on"):
            for survival in ("off", "on"):
                for loss in ("rvl", "safety"):
                    members[f"context_{context}_survival_{survival}_{loss}"] = {
                        seed: {"payload": {"perPose": []}, "safeWorkpoint": True}
                        for seed in (1, 2, 3)
                    }
        contrasts = _factor_contrasts(members)
        self.assertIn("context_main_effect", contrasts)
        self.assertEqual(len(contrasts["context_main_effect"]), 8)

    def test_factor_validation_safety_uses_variant_names(self) -> None:
        members = {
            "context_off_survival_off_rvl": {
                seed: {
                    "payload": {
                        "poseMacro": {"weightedRecall": 0.995},
                        "aggregate": {"weightedRecall": 0.996},
                    }
                }
                for seed in (1, 2, 3)
            },
            "context_on_survival_off_rvl": {
                seed: {
                    "payload": {
                        "poseMacro": {"weightedRecall": 0.998},
                        "aggregate": {"weightedRecall": 0.997},
                    }
                }
                for seed in (1, 2, 3)
            },
        }
        contrast = {
            "context_on_survival_off_rvl": (1.0, members["context_on_survival_off_rvl"]),
            "context_off_survival_off_rvl": (-1.0, members["context_off_survival_off_rvl"]),
        }
        self.assertEqual(
            _validation_member_safety(members, contrast),
            {
                "context_on_survival_off_rvl": True,
                "context_off_survival_off_rvl": True,
            },
        )

    def test_vectorized_bootstrap_keeps_paired_zero_contrast(self) -> None:
        def row(pose: int) -> dict[str, object]:
            return {
                "poseIndex": pose,
                "tp": 2, "fp": 1, "fn": 1, "tn": 6,
                "weightedTp": 2.0, "weightedGt": 3.0,
                "precision": 2.0 / 3.0, "recall": 2.0 / 3.0,
                "weightedRecall": 2.0 / 3.0, "accuracy": 8.0 / 10.0,
                "balancedAccuracy": 0.75, "specificity": 0.8,
                "f1": 2.0 / 3.0, "jaccard": 0.5,
                "usefulCull": 0.6, "badCull": 0.1,
                "predCount": 3, "predOverCandidate": 0.3, "predOverGt": 1.5,
                "candidateGlbCount": 2, "predictedGlbCount": 1,
                "candidateGlbBytes": 20.0, "predictedGlbBytes": 10.0,
                "glbCountReduction": 0.5, "glbByteReduction": 0.5,
                "downloadUtilityTp": 2.0,
                "glbBytesAtAchievedVisualUtility": 10.0,
                "candidateIdSha256": str(pose),
                "imageMetrics": {
                    "totalPixels": 100, "validReferencePixels": 100,
                    "PER": 0.01, "missPixelRate": 0.005,
                    "wrongInstancePixelRate": 0.003,
                    "extraPixelRateOverImage": 0.002,
                    "errorPixels": 1, "missPixels": 0.5,
                    "wrongInstancePixels": 0.3, "extraPixels": 0.2,
                },
            }

        member = {
            seed: {"payload": {"perPose": [row(pose) for pose in range(4)], "safeWorkpoint": True}}
            for seed in (1, 2, 3)
        }
        for metric in ("aggregate_recall", "aggregate_image_per"):
            result = _bootstrap_effect(member, member, metric, 100, 7)
            self.assertEqual(result["meanDelta"], 0.0)
            self.assertLessEqual(result["ci95"][0], 0.0)
            self.assertGreaterEqual(result["ci95"][1], 0.0)


if __name__ == "__main__":
    unittest.main()
