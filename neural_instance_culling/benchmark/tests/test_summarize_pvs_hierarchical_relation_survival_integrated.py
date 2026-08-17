from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

BENCHMARK = Path(__file__).resolve().parents[1]
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))

import summarize_pvs_hierarchical_relation_survival_integrated as summary  # noqa: E402


class SummarizeHierarchicalRelationTest(unittest.TestCase):
    @staticmethod
    def _row(pose: int, pred: int, candidate: int, gt: int) -> dict:
        tp = min(pred, gt)
        fp = pred - tp
        fn = gt - tp
        tn = candidate - tp - fp - fn
        return {
            "poseIndex": pose,
            "metrics": {
                "precision": tp / max(1, pred),
                "recall": tp / max(1, gt),
                "weightedRecall": 1.0,
                "f1": 2 * tp / max(1, pred + gt),
                "jaccard": tp / max(1, tp + fp + fn),
                "accuracy": (tp + tn) / candidate,
                "balancedAccuracy": 0.5,
                "specificity": tn / max(1, tn + fp),
                "usefulCull": tn / candidate,
                "badCull": fn / candidate,
                "avgPredCount": pred,
                "avgCandidateCount": candidate,
                "avgGtCount": gt,
                "predOverCandidate": pred / candidate,
                "predOverGt": pred / max(1, gt),
                "predictedGlbCount": 1.0,
                "candidateGlbCount": 2.0,
                "candidateGlbBytes": 20.0,
                "predictedGlbBytes": 10.0,
                "glbCountReduction": 0.5,
                "requiredGlbCount": 1.0,
                "requiredGlbBytes": 10.0,
                "glbByteReduction": 0.5,
                "downloadUtilityRecall": 1.0,
                "glbBytesAtAchievedVisualUtility": 10.0,
                "downloadNdcgAt20": 0.5,
                "downloadNdcgAt50": 0.5,
                "downloadNdcgAt100": 0.5,
                "utilityNdcgAt20": 0.5,
                "utilityNdcgAt50": 0.5,
                "utilityNdcgAt100": 0.5,
                "visibilityBaselineNdcgAt20": 0.5,
                "visibilityBaselineNdcgAt50": 0.5,
                "visibilityBaselineNdcgAt100": 0.5,
                "downloadUtilityRecallFraction5pct": 0.5,
                "downloadUtilityRecallFraction10pct": 0.5,
                "downloadUtilityRecallFraction20pct": 0.5,
                "downloadUtilityRecallFraction40pct": 0.5,
                "downloadUtilityRecallFixed10MiB": 0.5,
                "downloadUtilityRecallFixed25MiB": 0.5,
                "downloadUtilityRecallFixed50MiB": 0.5,
                "downloadRequiredGlbRecallAt100": 0.5,
                "missPixelRate": 0.0,
                "wrongIdPixelRate": 0.0,
                "extraPixelRate": 0.0,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "weightedTp": float(gt),
                "weightedGt": float(gt),
                "candidateCount": candidate,
            },
        }

    def test_pooled_counts_keep_average_count_semantics(self) -> None:
        rows = [self._row(0, pred=2, candidate=10, gt=1), self._row(1, pred=6, candidate=20, gt=2)]
        result = summary._summarize_rows(rows)
        aggregate = result["aggregate"]
        self.assertEqual(aggregate["avgPredCount"], 4.0)
        self.assertEqual(aggregate["avgCandidateCount"], 15.0)
        self.assertEqual(aggregate["avgGtCount"], 1.5)
        self.assertEqual(aggregate["tp"], 1.5)

    def test_seed_clustered_gather_has_expected_shape(self) -> None:
        arrays = [{"value": np.arange(5.0)}, {"value": np.arange(5.0) + 10.0}]
        selected_seed = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
        pose_indices = np.asarray(
            [[[0, 2, 4], [1, 3, 0]], [[4, 3, 2], [0, 1, 2]]],
            dtype=np.int64,
        )
        gathered = summary._gather_pose_field(arrays, "value", selected_seed, pose_indices)
        self.assertEqual(gathered.shape, (2, 2, 3))
        np.testing.assert_array_equal(gathered[0, 0], [0.0, 2.0, 4.0])
        np.testing.assert_array_equal(gathered[0, 1], [11.0, 13.0, 10.0])

    def test_zero_weight_pose_is_excluded_from_weighted_lcb_without_index_error(self) -> None:
        rows = [self._row(0, pred=2, candidate=10, gt=1), self._row(1, pred=3, candidate=10, gt=1)]
        rows[0]["metrics"]["weightedTp"] = 0.0
        rows[0]["metrics"]["weightedGt"] = 0.0
        result = summary._summarize_rows(rows, lcb_replicates=128, lcb_seed=7)
        self.assertTrue(np.isfinite(result["aggregate"]["weightedRecallLowerConfidenceBound"]))

    def test_glb_count_metrics_are_retained_in_aggregate_and_bootstrap_inputs(self) -> None:
        rows = [self._row(0, pred=2, candidate=10, gt=1), self._row(1, pred=3, candidate=10, gt=1)]
        rows[0]["metrics"].update({"candidateGlbCount": 5.0, "glbCountReduction": 0.4})
        rows[1]["metrics"].update({"candidateGlbCount": 7.0, "glbCountReduction": 0.6})
        result = summary._summarize_rows(rows)
        self.assertEqual(result["aggregate"]["candidateGlbCount"], 6.0)
        self.assertEqual(result["aggregate"]["glbCountReduction"], 0.5)

    def test_threshold_provenance_requires_calibration_and_no_test_selection(self) -> None:
        payload = {
            "threshold": 0.25,
            "thresholdSource": {
                "protocol": "calibration_ready_pre_test",
                "selectionSplit": "calibration",
                "selectedFromTest": False,
                "testEvaluationCount": 0,
                "selectedThreshold": 0.25,
                "safeWorkpoint": True,
            },
        }
        summary._validate_threshold_provenance(payload, Path("fixture.json"))
        payload["thresholdSource"]["selectionSplit"] = "validation"
        with self.assertRaises(ValueError):
            summary._validate_threshold_provenance(payload, Path("fixture.json"))

    def test_formal_comparisons_are_named_full_minus_ablation(self) -> None:
        pairs = summary._comparison_pairs([
            "full",
            "without_hierarchical_relation",
            "without_viewcell_integration",
        ])
        self.assertEqual(
            pairs,
            [
                ("full_minus_without_hierarchical_relation", "without_hierarchical_relation", "full"),
                ("full_minus_without_viewcell_integration", "without_viewcell_integration", "full"),
                ("without_viewcell_integration_minus_without_hierarchical_relation", "without_hierarchical_relation", "without_viewcell_integration"),
            ],
        )

    def test_paired_bootstrap_requires_shared_identity_and_three_seed_clusters(self) -> None:
        members = {}
        for variant in ("full", "without_hierarchical_relation"):
            for seed in (20260801, 20260802, 20260803):
                members[(variant, seed)] = {
                    "rows": [self._row(0, 2, 10, 1), self._row(1, 2, 10, 1)],
                    "poseIndices": [0, 1],
                    "candidateDigest": "a" * 64,
                }
        effects = summary._paired_bootstrap(
            members,
            "without_hierarchical_relation",
            "full",
            [20260801, 20260802, 20260803],
            ("precision",),
            10000,
            17,
        )
        self.assertEqual(effects["aggregate.precision"]["bootstrapReplicates"], 10000)
        members[("full", 20260803)]["candidateDigest"] = "b" * 64
        with self.assertRaises(ValueError):
            summary._paired_bootstrap(
                members,
                "without_hierarchical_relation",
                "full",
                [20260801, 20260802, 20260803],
                ("precision",),
                10000,
                17,
            )


if __name__ == "__main__":
    unittest.main()
