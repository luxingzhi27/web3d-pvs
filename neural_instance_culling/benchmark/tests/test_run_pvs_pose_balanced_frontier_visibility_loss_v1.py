from __future__ import annotations

from pathlib import Path
import unittest

from neural_instance_culling.benchmark.run_pvs_pose_balanced_frontier_visibility_loss_v1 import (
    FORMAL_EPOCHS,
    FORMAL_SEEDS,
    ROUND1_CONFIGS,
    SCAN_EPOCHS,
    _formal_specs,
    _rank,
    _round2_configs,
    build_train_command,
)


class PoseBalancedFrontierRunnerTests(unittest.TestCase):
    def test_scan_matrix_and_formal_contract_are_complete(self) -> None:
        self.assertEqual(len(ROUND1_CONFIGS), 4)
        self.assertEqual(len(_round2_configs(0.2)), 4)
        self.assertEqual(SCAN_EPOCHS, 12)
        self.assertEqual(FORMAL_EPOCHS, 40)
        specs = _formal_specs(ROUND1_CONFIGS[2])
        self.assertEqual(len(specs), 3)
        self.assertEqual({spec[2] for spec in specs}, set(FORMAL_SEEDS))

    def test_train_command_is_from_scratch_and_disables_old_visibility_terms(self) -> None:
        command = build_train_command(
            Path("/shared"),
            Path("/output/member"),
            ROUND1_CONFIGS[2],
            seed=FORMAL_SEEDS[0],
            epochs=FORMAL_EPOCHS,
        )
        joined = " ".join(command)
        self.assertNotIn("--initial-checkpoint", command)
        self.assertIn("--loss-variant pose_balanced_frontier", joined)
        self.assertIn("--utility-loss-weight 0.0", joined)
        self.assertIn("--download-loss-weight 0.0", joined)
        self.assertIn("--boundary-tail-weight 0.0", joined)
        self.assertIn("--negative-band-weight 0.0", joined)
        self.assertIn("--rvl-count-weight 0.0", joined)
        self.assertIn("--rvl-rank-weight 0.0", joined)
        self.assertIn("--epochs 40", joined)

    def test_safe_rows_rank_before_unsafe_rows(self) -> None:
        def row(*, safe: bool, precision: float, weighted_lcb: float):
            return {
                "eligibleSafe": safe,
                "aggregate": {
                    "precision": precision,
                    "weightedRecall": 1.0,
                    "weightedRecallLowerConfidenceBound": weighted_lcb,
                    "balancedAccuracy": 0.7,
                    "accuracy": 0.8,
                    "usefulCull": 0.6,
                    "avgPredCount": 1000.0,
                },
            }

        self.assertGreater(
            _rank(row(safe=True, precision=0.2, weighted_lcb=0.991)),
            _rank(row(safe=False, precision=0.9, weighted_lcb=0.989)),
        )


if __name__ == "__main__":
    unittest.main()
