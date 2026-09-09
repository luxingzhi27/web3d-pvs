from __future__ import annotations

from pathlib import Path
import sys
import unittest

BENCHMARK = Path(__file__).resolve().parents[1]
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))

from aabb_ray_baseline_config import (  # noqa: E402
    FORMAL_EPOCHS,
    FORMAL_SEEDS,
    FORMAL_STEPS_PER_EPOCH,
    LOSS_CONFIG,
    SCAN_EPOCHS,
    SCAN_LEARNING_RATES,
    SCAN_STEPS_PER_EPOCH,
)
from run_aabb_ray_baseline import (  # noqa: E402
    _aggregate_cull_rates,
    _row_key,
    _selected_threshold,
    build_train_command,
    formal_test_allowed,
)


class AabbRayFormalRunnerTests(unittest.TestCase):
    def test_aggregate_cull_rates_are_derived_from_confusion_counts(self) -> None:
        useful, bad = _aggregate_cull_rates({"tp": 5, "fp": 3, "fn": 2, "tn": 10})
        self.assertAlmostEqual(useful, 0.5)
        self.assertAlmostEqual(bad, 0.1)

    def test_safe_pool_ranks_useful_cull_before_recall_lcb(self) -> None:
        common = {
            "aggregateWeightedRecall": 0.995,
            "agg_balanced_accuracy": 0.9,
            "agg_specificity": 0.9,
            "agg_precision": 0.2,
            "avg_pred_count": 100,
        }
        more_cull = {
            **common,
            "aggregateWeightedRecallLowerConfidenceBound": 0.991,
            "tp": 10,
            "fp": 10,
            "fn": 0,
            "tn": 80,
        }
        higher_lcb = {
            **common,
            "aggregateWeightedRecallLowerConfidenceBound": 0.999,
            "tp": 10,
            "fp": 30,
            "fn": 0,
            "tn": 60,
        }
        self.assertGreater(_row_key(more_cull), _row_key(higher_lcb))
        self.assertGreater(
            _row_key(higher_lcb, safe_first=False),
            _row_key(more_cull, safe_first=False),
        )

    def test_registered_scan_and_confirmation_matrix_is_fixed(self) -> None:
        self.assertEqual(SCAN_LEARNING_RATES, (2e-4, 1e-3))
        self.assertEqual((SCAN_EPOCHS, SCAN_STEPS_PER_EPOCH), (6, 300))
        self.assertEqual((FORMAL_EPOCHS, FORMAL_STEPS_PER_EPOCH), (40, 900))
        self.assertEqual(FORMAL_SEEDS, (20260801, 20260802, 20260803))
        self.assertEqual(LOSS_CONFIG["lossVariant"], "pose_balanced_rvl_contrastive")
        self.assertEqual(LOSS_CONFIG["rvlRecallGuardWeight"], 0.30)
        self.assertEqual(LOSS_CONFIG["sharedTailSeparationWeight"], 0.20)

    def test_training_command_contains_mainline_objective_and_no_test_split(self) -> None:
        command = build_train_command(
            Path("/mnt/sda/rhyang/slm"),
            Path("/mnt/sda/rhyang/slm-wt-baselines/neural_instance_culling/model/out/fixture"),
            scene="hkust_v3",
            learning_rate=2e-4,
            seed=FORMAL_SEEDS[0],
            epochs=SCAN_EPOCHS,
            steps_per_epoch=SCAN_STEPS_PER_EPOCH,
        )
        self.assertIn("--loss-variant", command)
        self.assertEqual(command[command.index("--loss-variant") + 1], "pose_balanced_rvl_contrastive")
        self.assertIn("--integrated-rvl-recall-guard-weight", command)
        self.assertIn("--integrated-separation-weight", command)
        self.assertIn("--integrated-tail-ramp-fraction", command)
        self.assertNotIn("--split", command)
        self.assertNotIn("test", command)

    def test_formal_test_requires_calibration_frozen_threshold(self) -> None:
        self.assertFalse(formal_test_allowed(split="validation", checkpoint=None, calibration=None, threshold=None))
        self.assertFalse(formal_test_allowed(split="test", checkpoint=Path("model.pt"), calibration=None, threshold=0.5))
        self.assertFalse(formal_test_allowed(split="test", checkpoint=Path("model.pt"), calibration=Path("calibration.json"), threshold=None))
        self.assertTrue(formal_test_allowed(split="test", checkpoint=Path("model.pt"), calibration=Path("calibration.json"), threshold=0.5))

    def test_formal_test_threshold_reader_does_not_fall_back_to_diagnostic(self) -> None:
        safe = {
            "status": "safe",
            "bestSafe": {"selection": {"threshold": 0.7}},
            "bestDiagnostic": {"selection": {"threshold": 0.2}},
        }
        self.assertEqual(_selected_threshold(safe, require_safe=True), 0.7)
        with self.assertRaisesRegex(ValueError, "safe calibration"):
            _selected_threshold(
                {"status": "no_qualified_safety_workpoint", "bestDiagnostic": safe["bestDiagnostic"]},
                require_safe=True,
            )


if __name__ == "__main__":
    unittest.main()
