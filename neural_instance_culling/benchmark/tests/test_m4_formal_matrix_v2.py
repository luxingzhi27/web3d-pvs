from __future__ import annotations

import sys
import unittest
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_DIR))

from decide_m4_route_v2 import self_test as route_self_test  # noqa: E402
from m4_formal_matrix_v2_utils import (  # noqa: E402
    FACTOR_VARIANTS,
    SEEDS,
    bootstrap_effects_by_scope,
    candidate_digest,
    summarize_rows,
    validate_candidate_matrix,
)
from select_m4_v2_calibration_workpoints import self_test as calibration_self_test  # noqa: E402
from summarize_formal_m4_matrix_v2 import self_test as summary_self_test  # noqa: E402


def fixture_row(pose: int, tp: int, fp: int, fn: int, tn: int) -> dict:
    candidate = tp + fp + fn + tn
    gt = tp + fn
    recall = tp / max(1, gt)
    precision = tp / max(1, tp + fp)
    specificity = tn / max(1, tn + fp)
    return {
        "pose_index": pose,
        "candidate_id_sha256": f"{pose:064x}",
        "candidate_count": candidate,
        "gt_count": gt,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "weighted_recall": recall,
        "f1": 2 * precision * recall / max(1e-8, precision + recall),
        "jaccard": tp / max(1, tp + fp + fn),
        "accuracy": (tp + tn) / candidate,
        "balanced_accuracy": (recall + specificity) / 2,
        "specificity": specificity,
        "useful_cull": tn / candidate,
        "bad_cull": fn / candidate,
        "avg_pred_count": tp + fp,
        "weighted_tp": float(tp),
        "weighted_gt": float(gt),
    }


class M4FormalMatrixV2Tests(unittest.TestCase):
    def test_summary_calibration_route_self_tests(self) -> None:
        self.assertEqual(summary_self_test()["status"], "passed")
        self.assertEqual(calibration_self_test()["status"], "passed")
        self.assertEqual(route_self_test()["status"], "passed")

    def test_candidate_confusion_closure_and_scopes(self) -> None:
        rows = {0: fixture_row(0, 1, 1, 0, 2), 1: fixture_row(1, 1, 0, 1, 2)}
        summary = summarize_rows(rows.values(), lcb_replicates=100, lcb_seed=3)
        self.assertEqual(summary["aggregate"]["avg_candidate_count"], 4.0)
        self.assertAlmostEqual(summary["aggregate"]["accuracy"], 0.75)
        self.assertFalse(summary["safety"]["qualifiedSafetyWorkpoint"])

    def test_all_registered_members_are_paired_before_bootstrap(self) -> None:
        members = {}
        for variant in FACTOR_VARIANTS:
            for seed in SEEDS:
                members[(variant, seed)] = {"rows": {0: fixture_row(0, 1, 0, 0, 3), 1: fixture_row(1, 1, 0, 0, 3)}}
        identity = validate_candidate_matrix(members)
        self.assertEqual(identity["poseCount"], 2)
        self.assertEqual(len(identity["candidateDigest"]), 64)
        effects = bootstrap_effects_by_scope(
            members,
            {"full": 1.0, "geometry_context_ray": -1.0},
            ("precision", "recall", "weighted_recall", "accuracy", "balanced_accuracy", "f1", "useful_cull", "bad_cull", "avg_pred_count"),
            replicates=10000,
            seed=4,
        )
        self.assertEqual(effects["aggregate"]["recall"]["bootstrap_replicates"], 10000)

    def test_candidate_digest_is_stable(self) -> None:
        identity = {0: ("a" * 64, 3), 1: ("b" * 64, 4)}
        self.assertEqual(candidate_digest(identity), candidate_digest(dict(reversed(list(identity.items())))))


if __name__ == "__main__":
    unittest.main()
