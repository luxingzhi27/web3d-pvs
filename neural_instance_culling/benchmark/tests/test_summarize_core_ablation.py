from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from neural_instance_culling.benchmark.summarize_core_ablation import (
    COUNT_FIELDS,
    POSE_COUNT,
    SEEDS,
    _evaluation_path,
    _full_reference_path,
    _validate_pair,
    paired_hierarchical_bootstrap,
)


def _arrays(*, fp: float, tn: float, pred: float) -> dict[str, np.ndarray]:
    values = {
        "tp": 10.0,
        "fp": fp,
        "fn": 0.0,
        "tn": tn,
        "weightedTp": 100.0,
        "weightedGt": 100.0,
        "predCount": pred,
        "candidateGlbCount": 50.0,
        "predictedGlbCount": pred,
        "candidateGlbBytes": 1000.0,
        "predictedGlbBytes": pred * 10.0,
        "glbCountReduction": 1.0 - pred / 50.0,
        "glbByteReduction": 1.0 - pred / 100.0,
        "glbBytesAtAchievedVisualUtility": pred * 8.0,
    }
    self_check = set(values) == set(COUNT_FIELDS)
    if not self_check:
        raise AssertionError("test fixture drifted from bootstrap fields")
    return {
        field: np.full(POSE_COUNT, value, dtype=np.float64)
        for field, value in values.items()
    }


class CoreAblationSummaryTest(unittest.TestCase):
    def test_all_members_use_the_canonical_paper_directory(self) -> None:
        root = Path("/paper")
        self.assertEqual(
            _full_reference_path(root, SEEDS[0]),
            root / "members/paper_full_seed20260801_e40/validation_evaluation.json",
        )
        self.assertEqual(
            _evaluation_path(root, "no_relation", SEEDS[0]),
            root
            / "members/paper_no_relation_seed20260801_e40/validation_evaluation.json",
        )

    def test_paired_bootstrap_reports_full_minus_ablation_direction(self) -> None:
        full = {seed: _arrays(fp=2.0, tn=88.0, pred=12.0) for seed in SEEDS}
        ablation = {seed: _arrays(fp=10.0, tn=80.0, pred=20.0) for seed in SEEDS}

        result = paired_hierarchical_bootstrap(
            full, ablation, replicates=100, seed=20260823
        )

        self.assertGreater(result["posePrecision"]["ci95"][0], 0.0)
        self.assertGreater(result["aggregatePrecision"]["ci95"][0], 0.0)
        self.assertGreater(result["poseBalancedAccuracy"]["ci95"][0], 0.0)
        self.assertGreater(result["aggregateUsefulCull"]["ci95"][0], 0.0)
        self.assertLess(result["avgPredCount"]["ci95"][1], 0.0)
        self.assertEqual(result["poseWeightedRecall"]["ci95"], [0.0, 0.0])
        self.assertEqual(result["aggregateWeightedRecall"]["ci95"], [0.0, 0.0])

    def test_pair_validation_rejects_candidate_identity_drift(self) -> None:
        reference = [
            {"poseIndex": 1, "candidateCount": 2, "candidateIds": [3, 4]}
        ]
        variant = [
            {"poseIndex": 1, "candidateCount": 2, "candidateIds": [3, 5]}
        ]
        with self.assertRaisesRegex(ValueError, "candidate IDs differ"):
            _validate_pair(reference, variant, "fixture")


if __name__ == "__main__":
    unittest.main()
